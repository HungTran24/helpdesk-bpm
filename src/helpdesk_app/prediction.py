"""Dự đoán ticket đang chạy: bước tiếp theo và số ngày còn lại (baseline, Gradient Boosting, LSTM, Transformer),
và Decision Tree tại điểm rẽ sau Take in charge ticket. Dùng chung cho notebook và app.

Chia dữ liệu: xếp ticket theo ngày mở; 80% đầu để học, 20% cuối để kiểm tra; bỏ ticket vắt qua mốc.
"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from helpdesk_app.bpmn_embedding import node2vec
from helpdesk_app.event_log import ACTIVITY, CASE, RESOURCE, TIMESTAMP
from helpdesk_app.process_model import _pm4py_log, bpmn_graph

DAY = pd.Timedelta(days=1)
END = "Kết thúc"
MODELS = ["Baseline", "Gradient Boosting", "LSTM", "Transformer"]
MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "predictors.pt"


def split_by_time(event_log: pd.DataFrame, train_share: float = 0.8) -> tuple[pd.Index, pd.Index, pd.Timestamp]:
    span = event_log.groupby(CASE)[TIMESTAMP].agg(["min", "max"]).sort_values("min")
    cut = span["min"].iloc[int(len(span) * train_share)]
    return span.index[span["max"] < cut], span.index[span["min"] >= cut], cut


# ---------- Decision Tree tại điểm rẽ sau Take in charge ticket

TAKE_IN_CHARGE = "Take in charge ticket"
RESOLVE = "Resolve ticket"
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
CATEGORICAL, NUMERIC = ["resource", "weekday"], ["hour", "steps_before", "log_hours_before"]


def build_samples(event_log: pd.DataFrame) -> pd.DataFrame:
    """Mỗi ticket có Take in charge ticket thành một dòng, chụp tại lần nhận đầu tiên; `waits_later` là đáp án."""
    rows = []
    for case_id, events in event_log.groupby(CASE, sort=False):
        activities = events[ACTIVITY].tolist()
        if TAKE_IN_CHARGE not in activities:
            continue
        i = activities.index(TAKE_IN_CHARGE)
        taken_at = events[TIMESTAMP].iloc[i]
        rows.append({CASE: case_id, "resource": events[RESOURCE].iloc[i], "weekday": WEEKDAYS[taken_at.weekday()],
                     "hour": taken_at.hour, "steps_before": i,
                     "log_hours_before": np.log1p((taken_at - events[TIMESTAMP].iloc[0]).total_seconds() / 3600),
                     "waits_later": "Wait" in activities[i + 1:]})
    return pd.DataFrame(rows)


class WaitTree:
    """Decision Tree độ sâu 3: đọc luật "nếu… thì…" về việc rẽ vào Wait."""

    def __init__(self, train: pd.DataFrame):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=30).fit(train[CATEGORICAL])
        self.names = list(self.encoder.get_feature_names_out(CATEGORICAL)) + NUMERIC
        self.tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=0).fit(
            self.features(train), train["waits_later"])
        leaves = pd.Series(self.tree.apply(self.features(train)))
        self.leaf_wait = train["waits_later"].groupby(leaves.to_numpy()).mean()
        self.leaf_size = leaves.value_counts()

    def features(self, samples: pd.DataFrame) -> np.ndarray:
        return np.hstack([self.encoder.transform(samples[CATEGORICAL]), samples[NUMERIC].to_numpy()])

    def rule(self, sample: pd.DataFrame) -> tuple[list[str], float, int]:
        """Các điều kiện ticket đi qua, tỷ lệ Wait và số ticket của lá đó trong tập học."""
        x = self.features(sample)
        t, node, conditions = self.tree.tree_, 0, []
        while t.children_left[node] != -1:
            name, threshold = self.names[t.feature[node]], t.threshold[node]
            left = x[0, t.feature[node]] <= threshold
            conditions.append(_describe(name, threshold, left))
            node = t.children_left[node] if left else t.children_right[node]
        return conditions, float(self.leaf_wait[node]), int(self.leaf_size[node])


def _describe(name: str, threshold: float, left: bool) -> str:
    if name.startswith("resource_"):
        return f"người nhận {'không phải' if left else 'là'} {name.removeprefix('resource_')}"
    if name.startswith("weekday_"):
        return f"{'không nhận vào' if left else 'nhận vào'} {name.removeprefix('weekday_')}"
    if name == "log_hours_before":
        hours = np.expm1(threshold)
        span = f"{hours:.1f} giờ" if hours < 24 else f"{hours / 24:.1f} ngày"
        return f"được nhận {'trong vòng' if left else 'sau hơn'} {span} từ lúc mở".replace(".", ",")
    if name == "hour":
        return f"nhận {'trước' if left else 'sau'} {threshold:.0f} giờ"
    return f"có {'không quá' if left else 'hơn'} {threshold:.0f} bước trước khi nhận"


# ---------- Bước tiếp theo và thời gian còn lại

def _encode(events: pd.DataFrame, idx: dict[str, int]):
    a = [idx[x] for x in events[ACTIVITY]]
    t = events[TIMESTAMP].tolist()
    d = [0.0] + [(t[i] - t[i - 1]) / DAY for i in range(1, len(t))]
    e = [(x - t[0]) / DAY for x in t]
    return a, np.log1p(d), np.log1p(e), t


def prefixes(event_log: pd.DataFrame, ids, idx: dict[str, int]) -> list[tuple]:
    """Mỗi ticket cắt thành nhiều điểm dự đoán (sau bước 1, 2, …):
    (các bước đã đi, log khoảng cách giữa 2 bước, log thời gian đã trôi, bước kế tiếp hoặc END, số ngày còn lại).

    Số ngày còn lại = đến lần Resolve ticket đầu tiên (giống mục 5); NaN khi ticket đã Resolve hoặc không bao giờ Resolve.
    Không đo đến Closed vì sau Resolve ticket có khoảng 30 ngày chờ đóng gần như cố định.
    """
    rows, end, resolve = [], len(idx), idx[RESOLVE]
    for _, events in event_log[event_log[CASE].isin(ids)].groupby(CASE, sort=False):
        a, d, e, t = _encode(events, idx)
        resolved_at = t[a.index(resolve)] if resolve in a else None
        for k in range(1, len(a) + 1):
            left = (resolved_at - t[k - 1]) / DAY if resolved_at is not None and resolve not in a[:k] else np.nan
            rows.append((a[:k], d[:k], e[:k], a[k] if k < len(a) else end, left))
    return rows


def flat(P: list[tuple], emb: np.ndarray) -> np.ndarray:
    """Gradient Boosting: một dòng số (đếm từng loại bước, vector của bước cuối, số bước, thời gian)."""
    A = len(emb)
    X = np.zeros((len(P), A + emb.shape[1] + 4))
    for i, (a, d, e, *_) in enumerate(P):
        for x in a:
            X[i, x] += 1
        X[i, A:A + emb.shape[1]] = emb[a[-1]]
        X[i, -4:] = [len(a), d[-1], e[-1], np.mean(d)]
    return X


def tensors(P: list[tuple]):
    """LSTM, Transformer: các bước đã đi dạng tensor (mã bước, 2 cột thời gian, độ dài thật)."""
    L = max(len(p[0]) for p in P)
    a = torch.zeros(len(P), L, dtype=torch.long)
    f = torch.zeros(len(P), L, 2)
    for i, (aa, d, e, *_) in enumerate(P):
        a[i, :len(aa)] = torch.tensor(aa)
        f[i, :len(aa), 0] = torch.tensor(d)
        f[i, :len(aa), 1] = torch.tensor(e)
    return a, f, torch.tensor([len(p[0]) for p in P])


class LSTMNet(nn.Module):
    """Đọc lần lượt từng bước; vector bước Node2Vec cố định."""

    def __init__(self, emb: np.ndarray, outputs: int):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True)
        self.rnn = nn.LSTM(emb.shape[1] + 2, 64, batch_first=True)
        self.head_next, self.head_time = nn.Linear(64, outputs), nn.Linear(64, 1)

    def forward(self, a, f, lens):
        x = nn.utils.rnn.pack_padded_sequence(torch.cat([self.emb(a), f], -1), lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.rnn(x)
        return self.head_next(h[-1]), self.head_time(h[-1]).squeeze(-1)


class TransformerNet(nn.Module):
    """Nhìn mọi bước cùng lúc (attention): 2 lớp, 4 head, 64 chiều; vector bước Node2Vec cố định, chiếu lên 64 chiều."""

    def __init__(self, emb: np.ndarray, outputs: int, d: int = 64):
        super().__init__()
        self.emb = nn.Sequential(nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True), nn.Linear(emb.shape[1], d))
        self.time, self.pos = nn.Linear(2, d), nn.Embedding(32, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, dropout=0.1, batch_first=True), 2)
        self.head_next, self.head_time = nn.Linear(d, outputs), nn.Linear(d, 1)

    def forward(self, a, f, lens):
        n = a.shape[1]
        pos = torch.arange(n)
        h = self.enc(self.emb(a) + self.time(f) + self.pos(pos), mask=torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1),
                     src_key_padding_mask=pos[None] >= lens[:, None])
        h = h[torch.arange(len(lens)), lens - 1]  # đầu ra tại bước thật cuối cùng
        return self.head_next(h), self.head_time(h).squeeze(-1)


def train_net(net: nn.Module, P: list[tuple], scale: float, seed: int, epochs: int = 30) -> nn.Module:
    """Adam 0,001, batch 128; loss = cross-entropy (bước tiếp theo) + L1 (thời gian còn lại / trung bình)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    a, f, lens = tensors(P)
    y, r = torch.tensor([p[3] for p in P]), torch.tensor([p[4] for p in P], dtype=torch.float32)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 128):
            b = perm[i:i + 128]
            p_next, p_time = net(a[b], f[b], lens[b])
            known = ~torch.isnan(r[b])  # chỉ tính sai số thời gian khi ticket chưa Resolve
            loss = nn.functional.cross_entropy(p_next, y[b]) + nn.functional.l1_loss(p_time[known], r[b][known] / scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


class Predictors:
    """Baseline và 3 mô hình đã học trên tập học; `predict` nhận các bước đã đi của nhiều ticket."""

    def __init__(self, event_log: pd.DataFrame, train_ids, seed: int = 0):
        self.acts = sorted(event_log[ACTIVITY].unique())
        self.idx = {a: i for i, a in enumerate(self.acts)}
        A = len(self.acts)
        P = prefixes(event_log, train_ids, self.idx)
        y = np.array([p[3] for p in P])
        r = np.array([p[4] for p in P])
        known = ~np.isnan(r)

        nxt, rem = defaultdict(Counter), defaultdict(list)
        for a, _, _, yy, rr in P:
            nxt[a[-1]][yy] += 1
            if not np.isnan(rr):
                rem[a[-1]].append(rr)
        self.base_next = {k: c.most_common(1)[0][0] for k, c in nxt.items()}
        self.base_time = {k: float(np.median(v)) for k, v in rem.items()}
        self.base_default = (Counter(y).most_common(1)[0][0], float(np.median(r[known])))

        vec = node2vec(bpmn_graph(pm4py.discover_bpmn_inductive(_pm4py_log(event_log))))
        emb = np.array([vec[a] for a in self.acts], dtype=np.float32)
        self.emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-6)

        X = flat(P, self.emb)
        self.gb_next = HistGradientBoostingClassifier(l2_regularization=1.0, random_state=seed, early_stopping=False).fit(X, y)
        self.gb_time = HistGradientBoostingRegressor(loss="absolute_error", random_state=seed, early_stopping=False).fit(X[known], r[known])

        self.scale = float(r[known].mean())
        torch.set_num_threads(8)
        self.nets = {name: train_net(net(self.emb, A + 1), P, self.scale, seed)
                     for name, net in [("LSTM", LSTMNet), ("Transformer", TransformerNet)]}

    def name(self, i: int) -> str:
        return self.acts[i] if i < len(self.acts) else END

    def predict(self, P: list[tuple]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Mỗi mô hình: (mã bước tiếp theo, số ngày còn lại) cho từng điểm dự đoán."""
        last = [p[0][-1] for p in P]
        out = {"Baseline": (np.array([self.base_next.get(k, self.base_default[0]) for k in last]),
                            np.array([self.base_time.get(k, self.base_default[1]) for k in last]))}
        X = flat(P, self.emb)
        out["Gradient Boosting"] = (self.gb_next.predict(X), self.gb_time.predict(X))
        T = tensors(P)
        with torch.no_grad():
            for name, net in self.nets.items():
                p_next, p_time = net(*T)
                out[name] = (p_next.argmax(1).numpy(), (p_time * self.scale).numpy())
        return out

    def predict_ticket(self, events: pd.DataFrame) -> pd.DataFrame:
        """Các bước đã đi của một ticket → bảng: mô hình, bước tiếp theo, số ngày còn lại."""
        a, d, e, _ = _encode(events, self.idx)
        out = self.predict([(a, d, e)])
        resolved = self.idx[RESOLVE] in a  # đã giải quyết thì không còn thời gian để đoán
        return pd.DataFrame([{"mô hình": m, "bước tiếp theo": self.name(int(nx[0])),
                              "còn lại đến Resolve (ngày)": np.nan if resolved else max(float(tm[0]), 0.0)}
                             for m, (nx, tm) in out.items()])

    def score(self, P: list[tuple]) -> pd.DataFrame:
        y = np.array([p[3] for p in P])
        r = np.array([p[4] for p in P])
        known = ~np.isnan(r)
        return pd.DataFrame([{"mô hình": m, "bước tiếp theo: đúng": (nx == y).mean(),
                              "thời gian còn lại: sai TB (ngày)": np.abs(tm[known] - r[known]).mean()}
                             for m, (nx, tm) in self.predict(P).items()]).set_index("mô hình")

    def at_gateway(self, P: list[tuple]) -> tuple[int, float, pd.Series]:
        """Điểm rẽ: ngay sau lần Take in charge ticket đầu tiên.
        Trả về (số điểm, tỷ lệ thực tế rẽ vào Wait, số lần mỗi mô hình đoán Wait)."""
        t, w = self.idx[TAKE_IN_CHARGE], self.idx["Wait"]
        G = [p for p in P if p[0][-1] == t and p[0].count(t) == 1]
        y = np.array([p[3] for p in G])
        return len(G), float((y == w).mean()), pd.Series({m: int((nx == w).sum()) for m, (nx, _) in self.predict(G).items()})


def remind_by_prediction(predictors: Predictors, event_log: pd.DataFrame, ids) -> pd.DataFrame:
    """Khi ticket vừa vào Wait, hẹn nhắc sau đúng số ngày mô hình đoán còn lại; so với hẹn cố định X ngày
    có cùng số lần nhắc. Chỉ lấy các lần Wait trước lần Resolve ticket đầu tiên. "Bớt" = số ngày chờ bị cắt khi nhắc (giả định nhắc xong đi tiếp ngay), chia đều cho mọi lần Wait."""
    P, waited = [], []
    for case_id, events in event_log[event_log[CASE].isin(ids)].groupby(CASE, sort=False):
        acts, times = events[ACTIVITY].tolist(), events[TIMESTAMP].tolist()
        for k, point in enumerate(prefixes(events, [case_id], predictors.idx), start=1):
            if acts[k - 1] == "Wait" and k < len(acts) and not np.isnan(point[4]):  # chỉ Wait trước lần Resolve đầu, như mục 6
                P.append(point)
                waited.append((times[k] - times[k - 1]) / DAY)
    waited = np.array(waited)

    def cut(deadline):
        late = waited > deadline
        return late.mean(), (waited - deadline)[late].sum() / len(waited)

    rows = []
    for model, (_, days) in predictors.predict(P).items():
        deadline = np.maximum(days, 0)
        share, saved = cut(deadline)
        fixed = np.quantile(waited, 1 - share)  # ngưỡng cố định cho cùng tỷ lệ lần Wait bị nhắc
        rows.append({"mô hình": model, "hẹn (trung vị ngày)": float(np.median(deadline)), "lần Wait bị nhắc": share,
                     "bớt / lần Wait (ngày)": saved, "cố định cùng số lần nhắc: X (ngày)": fixed,
                     "cố định: bớt / lần Wait (ngày)": cut(fixed)[1]})
    return pd.DataFrame(rows).set_index("mô hình").assign(waits=len(waited))


def load_or_train(event_log: pd.DataFrame, path: Path = MODEL_FILE) -> Predictors:
    """Lần đầu học rồi lưu (khoảng 1–2 phút); các lần sau đọc lại file."""
    if path.exists():
        return torch.load(path, weights_only=False)
    train_ids, _, _ = split_by_time(event_log)
    predictors = Predictors(event_log, train_ids)
    path.parent.mkdir(exist_ok=True)
    torch.save(predictors, path)
    return predictors


if __name__ == "__main__":
    from helpdesk_app import prediction  # lưu dưới tên module thật, không phải __main__, để app đọc lại được
    from helpdesk_app.event_log import load_event_log

    log = load_event_log()
    train_ids, test_ids, cut = split_by_time(log)
    predictors = prediction.load_or_train(log)
    print(predictors.score(prefixes(log, test_ids, predictors.idx)).round(4))
