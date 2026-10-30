"""Helpdesk Ticket Monitor: công cụ cho trưởng nhóm Helpdesk. Chỉ gọi module và hiển thị."""
import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from helpdesk_app.bpmn_embedding import cluster_nodes, graph2vec_similarity, node2vec
from helpdesk_app.event_log import ACTIVITY, CASE, RESOURCE, TIMESTAMP, load_event_log
from helpdesk_app.prediction import (END, MODELS, RESOLVE, TAKE_IN_CHARGE, WaitTree, build_samples, load_or_train, prefixes,
                                     remind_by_prediction,
                                     split_by_time)
from helpdesk_app.process_model import (REMINDER_STEP, STANDARD_PROCESS, bpmn_graph, compare_with_standard,
                                        discover_bpmn, graph_dot, standard_bpmn, to_be_graph)
from helpdesk_app.waiting import by_quarter, days_to_resolve, wait_durations, wait_share_by_year, waiting_at

WAIT_COLOR = "#f59f00"
EXTRA_COLOR = "#ffe8cc"
GREY = "#c5ccd6"
CLUSTER_COLORS = ["#cfe2ff", "#ffe5b4", "#d4f4dd"]
EVENT_COLUMNS = {ACTIVITY: "Bước", RESOURCE: "Người làm", TIMESTAMP: "Thời điểm"}

st.set_page_config(page_title="Helpdesk Ticket Monitor", layout="wide")


def vn(number: float, digits: int = 0) -> str:
    """Số kiểu Việt Nam: 4580 -> "4.580", 19.1 -> "19,1"."""
    return f"{number:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def describe_deviation(kind: str) -> str:
    """"thừa Wait" -> "Thêm bước Wait"; bước có trong chuẩn mà bị thừa nghĩa là làm lại."""
    action, step = kind.split(" ", 1)
    if action == "thiếu":
        return f"Bỏ qua {step}"
    return f"Làm lại {step}" if step in STANDARD_PROCESS else f"Thêm bước {step}"


def events_table(events: pd.DataFrame) -> pd.DataFrame:
    table = events[list(EVENT_COLUMNS)].rename(columns=EVENT_COLUMNS).reset_index(drop=True)
    table["Thời điểm"] = table["Thời điểm"].dt.strftime("%d/%m/%Y %H:%M")
    table.index += 1
    return table


def path_dot(steps: list[tuple[str, str]]) -> str:
    """Đường đi của một ticket: bước khớp chuẩn để trắng, bước thêm tô cam, bước bị bỏ qua viền đứt."""
    lines = ["digraph { rankdir=LR; node [fontname=Helvetica, fontsize=11, shape=box, style=\"rounded,filled\"];"]
    for i, (step, kind) in enumerate(steps):
        if kind == "khớp":
            lines.append(f'n{i} [label="{step}", fillcolor=white];')
        elif kind == "thừa":
            lines.append(f'n{i} [label="{step}\\n(thêm)", fillcolor="{WAIT_COLOR if step == "Wait" else EXTRA_COLOR}"];')
        else:
            lines.append(f'n{i} [label="{step}\\n(bỏ qua)", fillcolor=white, style="rounded,dashed", fontcolor=gray40, color=gray50];')
    lines += [f"n{i} -> n{i + 1};" for i in range(len(steps) - 1)]
    return "\n".join(lines + ["}"])


@st.cache_resource
def base_data():
    event_log = load_event_log()
    year = event_log.groupby(CASE)[TIMESTAMP].min().dt.year
    return event_log, year, compare_with_standard(event_log).set_index(CASE), bpmn_graph(standard_bpmn())


@st.cache_resource
def real_process(_event_log: pd.DataFrame, year: str):
    bpmn, share = discover_bpmn(_event_log)
    return bpmn_graph(bpmn), share


@st.cache_data
def year_similarity(_event_log: pd.DataFrame, _case_year: pd.Series) -> pd.DataFrame:
    graphs = {y: real_process(_event_log[_event_log[CASE].isin(_case_year[_case_year == y].index)], str(y))[0]
              for y in sorted(_case_year.unique())}
    return graph2vec_similarity(graphs)[0]


@st.cache_data
def reminder_effect(_event_log: pd.DataFrame) -> pd.DataFrame:
    rows = [{"nhắc khi chờ quá (ngày)": x, "trung vị (ngày)": days_to_resolve(_event_log, x).query("has_wait")["days"].median()}
            for x in range(1, 31)]
    return pd.DataFrame(rows)


@st.cache_resource
def prediction_data(_event_log: pd.DataFrame):
    train_ids, test_ids, cut = split_by_time(_event_log)
    predictors = load_or_train(_event_log)
    samples = build_samples(_event_log)
    tree = WaitTree(samples[samples[CASE].isin(train_ids)])
    test_points = prefixes(_event_log, test_ids, predictors.idx)
    resolve = days_to_resolve(_event_log)["days"]
    speed = (resolve[resolve.index.isin(train_ids)].median(), resolve[resolve.index.isin(test_ids)].median())
    return (predictors, tree, samples.set_index(CASE), test_ids, cut, predictors.score(test_points), predictors.at_gateway(test_points),
            speed, remind_by_prediction(predictors, _event_log, test_ids))


event_log, case_year, comparison, standard_graph = base_data()

page = st.sidebar.radio("Màn hình", ["1. Quy trình", "2. Theo dõi Wait", "3. Ticket đang chạy"])
st.sidebar.caption(f"Event log Helpdesk: {vn(len(comparison))} ticket, 2010–2014.")

if page.startswith("1"):
    year = st.segmented_control("Năm mở ticket", ["Tất cả", 2010, 2011, 2012, 2013], default="Tất cả")
    cases = case_year.index if year in (None, "Tất cả") else case_year[case_year == year].index
    chosen = comparison.loc[cases]
    fit_share = chosen["fits"].mean()
    kinds = chosen["deviations"].map(set).explode().dropna().value_counts()
    wait_share = kinds.get("thừa Wait", 0) / len(chosen)

    st.title(f"Chỉ {fit_share:.0%} ticket đi đúng quy trình chuẩn")
    st.caption("Kiến thức: BPMN · Process Mining · Petri Net")
    cols = st.columns(3)
    cols[0].metric("Số ticket", vn(len(chosen)))
    cols[1].metric("Đi đúng chuẩn", f"{fit_share:.0%}", help=f"{vn(chosen['fits'].sum())} ticket")
    cols[2].metric("Có thêm bước Wait", f"{wait_share:.0%}", help=f"{vn(kinds.get('thừa Wait', 0))} ticket")

    st.subheader("Quy trình chuẩn: 4 bước nối tiếp")
    st.graphviz_chart(graph_dot(standard_graph))
    real_graph, real_share = real_process(event_log[event_log[CASE].isin(cases)], str(year))
    st.subheader("Quy trình thực tế: có thêm nhánh Wait và các vòng làm lại")
    st.graphviz_chart(graph_dot(real_graph, {"Wait": WAIT_COLOR}))
    st.caption(f"Vẽ từ 5 đường đi phổ biến nhất, chiếm {real_share:.0%} số ticket. "
               "Mũi tên nối thẳng giữa hai ô X: ticket bỏ qua đoạn đó, không làm bước nào rồi đi tiếp.")

    st.subheader("Kiểu lệch chuẩn hay gặp")
    top = kinds.head(8).rename_axis("kind").reset_index(name="tickets")
    top["kiểu lệch"] = top["kind"].map(describe_deviation)
    top["tỷ lệ"] = top["tickets"] / len(chosen)
    st.altair_chart(alt.Chart(top).mark_bar().encode(
        x=alt.X("tickets:Q", title="Số ticket", axis=alt.Axis(format="d")),
        y=alt.Y("kiểu lệch:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13)),
        color=alt.condition(alt.datum.kind == "thừa Wait", alt.value(WAIT_COLOR), alt.value(GREY)),
        tooltip=["kiểu lệch", alt.Tooltip("tickets:Q", title="Số ticket"), alt.Tooltip("tỷ lệ:Q", format=".0%")]
    ).properties(height=320), width="stretch")

    st.subheader("Xem một ticket so với chuẩn")
    with_wait = chosen.index[chosen["deviations"].map(lambda d: "thừa Wait" in d)]
    options = list(chosen.index)
    case_id = st.selectbox("Ticket", options, index=options.index(with_wait[0]) if len(with_wait) else 0,
                           format_func=lambda c: f"{c} · " + ("đúng chuẩn" if comparison.at[c, "fits"] else
                                                               ", ".join(describe_deviation(d) for d in dict.fromkeys(comparison.at[c, "deviations"]))))
    st.graphviz_chart(path_dot(comparison.at[case_id, "steps"]))
    st.caption("Ô cam: bước thêm vào so với chuẩn. Ô viền đứt: bước của chuẩn bị bỏ qua.")
    st.dataframe(events_table(event_log[event_log[CASE] == case_id]), width="stretch")

    st.markdown("**Tiếp theo:** bước Wait làm ticket chậm bao lâu? Xem màn hình *2. Theo dõi Wait*.")

    with st.expander("Cách làm"):
        st.markdown("""
- **Quy trình chuẩn:** nhóm vẽ BPMN theo đường đi phổ biến nhất.
- **Quy trình thực tế:** dùng Inductive Miner dựng BPMN từ event log.
- **So với chuẩn:** chuyển BPMN chuẩn sang Petri Net, rồi đối chiếu từng ticket (alignment) để tìm bước thêm vào hoặc bị bỏ qua.
""")

    with st.expander("Các bước của quy trình chia thành 3 giai đoạn (Node2Vec + K-Means)"):
        vectors = node2vec(real_graph)
        clusters = cluster_nodes(vectors)
        pca = PCA(2, random_state=42)
        xy = pca.fit_transform([vectors[n] for n in clusters.index])
        points = pd.DataFrame({"PC1": xy[:, 0], "PC2": xy[:, 1], "cụm": clusters.astype(str).to_numpy(),
                               "bước": clusters.index, "nhãn": [n if not n.startswith("XOR") else "" for n in clusters.index]})
        color = alt.Color("cụm:N", scale=alt.Scale(domain=["0", "1", "2"], range=CLUSTER_COLORS), legend=None)
        base = alt.Chart(points).encode(x="PC1:Q", y="PC2:Q")
        # Tâm cụm của K-Means = trung bình vector các node trong cụm, chiếu cùng phép PCA.
        centers = pd.DataFrame(pca.transform(pd.DataFrame([vectors[n] for n in clusters.index]).groupby(clusters.to_numpy()).mean()),
                               columns=["PC1", "PC2"]).assign(cụm=lambda t: t.index.astype(str))
        st.altair_chart(base.mark_circle(size=400, stroke="black", strokeWidth=1, opacity=1).encode(color=color, tooltip=["bước"])
                        + alt.Chart(centers).mark_point(shape="cross", size=500, filled=True, stroke="black", strokeWidth=1,
                                                        opacity=1).encode(x="PC1:Q", y="PC2:Q", color=color)
                        # Start, End, Closed nằm sát bước khác hoặc tâm cụm nên đặt chữ chỗ khác để không đè nhau.
                        + base.mark_text(dy=-18, fontSize=13).encode(text="nhãn:N").transform_filter("indexof(['Start','End','Closed'], datum.bước) < 0")
                        + base.mark_text(dy=20, fontSize=13).encode(text="nhãn:N").transform_filter("datum.bước == 'Start' || datum.bước == 'End'")
                        + base.mark_text(dx=16, align="left", fontSize=13).encode(text="nhãn:N").transform_filter("datum.bước == 'Closed'"),
                        width="stretch")
        st.caption("Mỗi chấm là một bước của BPMN thực tế, dấu + là tâm cụm, chấm không tên là cổng rẽ nhánh.")
        st.graphviz_chart(graph_dot(real_graph, {n: CLUSTER_COLORS[c] for n, c in clusters.items()}))
        st.markdown("""
- **Node2Vec:** đi ngẫu nhiên trên BPMN, bước nào hay đi cạnh nhau thì có vector gần nhau.
- **K-Means:** chia các vector thành 3 cụm.
- **PCA:** giảm vector còn 2 chiều để vẽ.
""")

elif page.startswith("2"):
    waits = wait_durations(event_log)
    as_is = days_to_resolve(event_log)
    by_wait = as_is.groupby("has_wait")["days"].median()
    st.title(f"Ticket có Wait mất {vn(by_wait[True], 1)} ngày, gấp {by_wait[True] / by_wait[False]:.0f} lần ticket không có Wait")
    st.caption("Kiến thức: Đánh giá quy trình · Điểm nghẽn · BPR · Petri Net (soundness)")
    cols = st.columns(3)
    cols[0].metric("Mỗi lần Wait chờ", f"{vn(waits.median(), 1)} ngày")
    cols[1].metric("Ticket có Wait", f"{vn(by_wait[True], 1)} ngày")
    cols[2].metric("Ticket không có Wait", f"{vn(by_wait[False], 1)} ngày")
    st.caption("Trung vị số ngày từ bước đầu đến Resolve ticket. Dùng trung vị vì vài ticket kéo rất dài làm trung bình bị đẩy lên.")

    by_year = wait_share_by_year(event_log).reset_index()
    last_year = by_year.iloc[-1]
    st.subheader(f"Năm {last_year['year']:.0f}, {last_year['wait_share']:.0%} ticket rơi vào Wait")
    st.altair_chart(alt.Chart(by_year).mark_bar().encode(
        x=alt.X("year:O", title="Năm mở ticket", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("wait_share:Q", title="Tỷ lệ ticket có Wait", axis=alt.Axis(format=".0%")),
        color=alt.condition(alt.datum.year == int(last_year["year"]), alt.value(WAIT_COLOR), alt.value(GREY)),
        tooltip=[alt.Tooltip("year:O", title="Năm"), alt.Tooltip("wait_share:Q", title="Có Wait", format=".0%"),
                 alt.Tooltip("tickets:Q", title="Số ticket")]).properties(height=260), width="stretch")

    st.subheader("Quy trình 2013 không đổi, chỉ nhiều ticket rẽ vào Wait hơn")
    st.caption("Kiến thức: Graph2Vec")
    straight = comparison["fits"].groupby(case_year).mean()
    left, right = st.columns(2)
    left.markdown("**Độ giống BPMN giữa các năm** (1 = giống hệt)")
    left.dataframe(year_similarity(event_log, case_year).style.format("{:.2f}").background_gradient(cmap="Oranges", vmin=0, vmax=1),
                   width="stretch")
    right.metric("Đi thẳng 4 bước, năm 2012", f"{straight[2012]:.0%}")
    right.metric("Đi thẳng 4 bước, năm 2013", f"{straight[2013]:.0%}", f"{(straight[2013] - straight[2012]) * 100:.0f} điểm %")
    st.markdown("BPMN năm 2013 giống 2011 và 2012 như hai năm đó giống nhau: hình dạng quy trình không đổi, chỉ ít ticket đi thẳng hơn.")
    with st.expander("Cách làm"):
        st.markdown("""
- Mỗi năm dựng một BPMN thực tế (Inductive Miner).
- **Graph2Vec:** đổi tên mỗi bước theo các bước trước và sau nó, coi mỗi BPMN là một văn bản, rồi học vector bằng Doc2Vec.
- Độ giống = cosine giữa vector của hai năm, trung bình 5 lần học.
""")

    quarters = by_quarter(event_log)
    st.subheader("Nhưng từ quý 4/2012, mọi ticket đều chậm hơn")
    lines = quarters[["days_with_wait", "days_without_wait"]].rename(
        columns={"days_with_wait": "có Wait", "days_without_wait": "không có Wait"}).rename_axis("quý").reset_index().melt(
        "quý", var_name="ticket", value_name="ngày")
    st.altair_chart(alt.Chart(lines).mark_line(point=True).encode(
        x=alt.X("quý:O", title="Quý mở ticket", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("ngày:Q", title="Trung vị số ngày đến Resolve"),
        color=alt.Color("ticket:N", scale=alt.Scale(domain=["có Wait", "không có Wait"], range=[WAIT_COLOR, GREY]), title=None),
        tooltip=["quý", "ticket", alt.Tooltip("ngày:Q", format=".1f")]).properties(height=260), width="stretch")
    before, after = quarters.loc["2012Q3"], quarters.loc["2012Q4"]
    for col, (label, key) in zip(st.columns(3), [("Ticket không có Wait đến Resolve", "days_without_wait"),
                                                  ("Chờ người nhận (Assign → Take in charge)", "days_to_take"),
                                                  ("Chờ đóng (Resolve → Closed)", "days_to_close")]):
        col.metric(label, f"{vn(before[key], 1)} → {vn(after[key], 1)} ngày", help="Trung vị, quý 3/2012 → quý 4/2012")
    st.caption("Trung vị, quý 3/2012 → quý 4/2012.")
    new_era = event_log[CASE].map(event_log.groupby(CASE)[TIMESTAMP].min() >= pd.Timestamp("2012-10-01", tz="UTC"))
    resolved_waits = waits[event_log.loc[waits.index, CASE].isin(as_is.index)]  # như báo cáo 5.4: chỉ ticket có Resolve
    era = new_era[resolved_waits.index]
    wait_before, wait_after = resolved_waits[~era].median(), resolved_waits[era].median()
    st.markdown("Cùng lúc đó số ticket mỗi quý giảm và thời gian chờ đóng đổi từ khoảng 30 ngày xuống đúng 15 ngày: "
                "cách làm việc đã đổi. Log không cho biết vì sao.")
    st.markdown(f"Từ quý 4/2012, ticket có và không có Wait gần bằng nhau, nhưng vì hai lý do khác nhau: mỗi lần Wait dài hơn "
                f"({vn(wait_before, 1)} → {vn(wait_after, 1)} ngày), còn ticket không có Wait chậm vì chờ người nhận và xử lý lâu. "
                "**Wait là chỗ sửa được ngay: nhắc khi ticket chờ quá hạn.**")

    st.subheader("Cải tiến: nhắc khi ticket chờ quá hạn")
    st.graphviz_chart(graph_dot(to_be_graph(), {"Wait": WAIT_COLOR, REMINDER_STEP: EXTRA_COLOR}))
    st.caption("BPMN TO-BE đã chuyển sang Petri Net và kiểm tra: không có deadlock.")

    st.subheader("Ticket cần nhắc")
    left, right = st.columns(2)
    today = left.date_input("Hôm nay là ngày", dt.date(2013, 6, 3), min_value=dt.date(2010, 1, 13), max_value=dt.date(2014, 1, 3),
                            format="DD/MM/YYYY")
    remind_after = right.slider("Nhắc khi chờ quá (ngày)", 1, 30, 7)
    waiting = waiting_at(event_log, pd.Timestamp(today, tz="UTC"))
    overdue = waiting[waiting["days_waiting"] > remind_after]
    st.metric("Cần nhắc", f"{len(overdue)} ticket", help=f"{len(waiting)} ticket đang ở Wait vào ngày {today:%d/%m/%Y}")
    if overdue.empty:
        st.info("Không có ticket nào chờ quá hạn vào ngày này.")
    else:
        table = pd.DataFrame({"Ticket": overdue[CASE], "Chờ từ": overdue[TIMESTAMP].dt.strftime("%d/%m/%Y"),
                              "Đã chờ (ngày)": overdue["days_waiting"].round(1), "Người chuyển sang Wait": overdue[RESOURCE]})
        picked = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row")
        row = picked.selection.rows[0] if picked.selection.rows else 0
        case_id = table["Ticket"].iloc[row]
        st.markdown(f"**{case_id}**: các bước đến ngày {today:%d/%m/%Y} (chọn dòng khác trong bảng để xem ticket khác)")
        events = event_log[event_log[CASE] == case_id]
        st.dataframe(events_table(events[events[TIMESTAMP] <= pd.Timestamp(today, tz="UTC")]), width="stretch")

    st.subheader(f"Nếu nhắc sau {remind_after} ngày, ticket có Wait xong nhanh hơn bao nhiêu?")
    effect = reminder_effect(event_log)
    after = effect.loc[effect["nhắc khi chờ quá (ngày)"] == remind_after, "trung vị (ngày)"].iloc[0]
    cols = st.columns(3)
    cols[0].metric("Hiện tại", f"{vn(by_wait[True], 1)} ngày")
    cols[1].metric(f"Nhắc sau {remind_after} ngày", f"{vn(after, 1)} ngày", f"-{vn(by_wait[True] - after, 1)} ngày", delta_color="inverse")
    cols[2].metric("Lần Wait bị nhắc", f"{(waits > remind_after).mean():.0%}")
    line = alt.Chart(effect).mark_line(point=alt.OverlayMarkDef(color=GREY), color=GREY).encode(
        x=alt.X("nhắc khi chờ quá (ngày):Q", title="Nhắc khi chờ quá (ngày)"),
        y=alt.Y("trung vị (ngày):Q", title="Ticket có Wait: số ngày đến Resolve"),
        tooltip=[alt.Tooltip("nhắc khi chờ quá (ngày):Q", title="Nhắc sau"), alt.Tooltip("trung vị (ngày):Q", format=".1f", title="Số ngày")])
    mark = alt.Chart(effect[effect["nhắc khi chờ quá (ngày)"] == remind_after]).mark_point(size=200, filled=True, color=WAIT_COLOR).encode(
        x="nhắc khi chờ quá (ngày):Q", y="trung vị (ngày):Q")
    st.altair_chart((line + mark).properties(height=260), width="stretch")
    st.caption("Giả định: nhắc xong thì ticket đi tiếp ngay. Đây là mức nhanh nhất có thể; thực tế có lần nhắc rồi khách vẫn chưa trả lời.")

    with st.expander("Cách làm"):
        st.markdown("""
- **Thời gian chờ:** từ lúc ghi Wait đến bước kế tiếp (log chỉ ghi lúc mỗi bước xong).
- **Ticket đang chờ:** bước gần nhất trước ngày đã chọn là Wait. Đã chờ = ngày đã chọn − thời điểm ghi Wait.
- **Mô phỏng:** mỗi lần chờ dài hơn X ngày được cắt còn X ngày, các bước khác giữ nguyên, rồi tính lại số ngày đến Resolve ticket.
- **Kiểm tra TO-BE:** chuyển BPMN TO-BE sang Petri Net, kiểm tra soundness (luôn đi được đến kết thúc, không kẹt).
""")

else:
    st.title("Ticket đang chạy: bước tiếp theo là gì, còn bao lâu đến Resolve?")
    st.caption("Kiến thức: Decision Tree tại điểm rẽ · Node2Vec · Dự đoán (Gradient Boosting, LSTM, Transformer)")
    with st.spinner("Đang chuẩn bị mô hình (lần đầu mất 1–2 phút)…"):
        (predictors, tree, samples, test_ids, cut, scores, (gate_n, gate_wait, gate_guesses), (train_days, test_days),
         reminders) = prediction_data(event_log)

    left, right = st.columns(2)
    options = sorted(test_ids, key=lambda c: int(c.removeprefix("Case")))
    case_id = left.selectbox(f"Ticket (mở từ {cut:%d/%m/%Y}, mô hình chưa thấy khi học)", options,
                             index=options.index("Case65") if "Case65" in options else 0)
    events = event_log[event_log[CASE] == case_id]
    k = left.slider("Đã đi đến bước", 1, len(events), min(2, len(events)))
    seen, later = events.iloc[:k], events.iloc[k:]

    left.markdown("**Đã biết đến lúc này**")
    left.dataframe(events_table(seen), width="stretch")

    actual_next = later[ACTIVITY].iloc[0] if len(later) else END
    resolved = events[events[ACTIVITY] == RESOLVE]
    actual_left = (float("nan") if resolved.empty or RESOLVE in seen[ACTIVITY].values
                   else (resolved[TIMESTAMP].iloc[0] - seen[TIMESTAMP].iloc[-1]) / pd.Timedelta(days=1))
    guesses = predictors.predict_ticket(seen)
    table = pd.concat([guesses, pd.DataFrame([{"mô hình": "Thực tế", "bước tiếp theo": actual_next, "còn lại đến Resolve (ngày)": actual_left}])], ignore_index=True)
    right.markdown("**Các mô hình đoán**")
    right.dataframe(table.style.format({"còn lại đến Resolve (ngày)": "{:.1f}"}, na_rep="đã Resolve").apply(
        lambda r: ["font-weight: bold"] * len(r) if r["mô hình"] == "Thực tế" else
        ["background-color: #e6f4ea" if c == "bước tiếp theo" and r[c] == actual_next else "" for c in r.index], axis=1),
        hide_index=True, width="stretch")
    right.caption("Ô xanh: đoán đúng bước tiếp theo.")

    taken_first = TAKE_IN_CHARGE in seen[ACTIVITY].tolist() and seen[ACTIVITY].tolist().index(TAKE_IN_CHARGE) == k - 1
    if taken_first and case_id in samples.index:
        conditions, wait_rate, size = tree.rule(samples.loc[[case_id]])
        right.info(f"**Điểm rẽ sau Take in charge ticket.** Ticket này: {'; '.join(conditions)}. "
                   f"Trong tập học, {wait_rate:.0%} ticket như vậy rẽ vào Wait ({vn(size)} ticket).")
    with right.expander("Diễn biến thật sau đó"):
        st.dataframe(events_table(later).set_index(pd.RangeIndex(k + 1, len(events) + 1)) if len(later) else
                     pd.DataFrame({"Bước": [END]}), width="stretch")

    st.subheader(f"Trên {vn(len(test_ids))} ticket kiểm tra: bước tiếp theo đoán tốt hơn baseline, thời gian còn lại thì gần bằng")
    st.dataframe(scores.loc[MODELS].style.format({"bước tiếp theo: đúng": "{:.1%}", "thời gian còn lại: sai TB (ngày)": "{:.1f}"})
                 .highlight_max(subset="bước tiếp theo: đúng", color="#e6f4ea")
                 .highlight_min(subset="thời gian còn lại: sai TB (ngày)", color="#e6f4ea"), width="stretch")
    st.caption("Sai TB: trung bình |số ngày đoán − số ngày thực tế| đến Resolve ticket. Báo cáo dùng trung bình 3 lần học; app dùng 1 lần.")
    st.markdown(f"Thời gian khó đoán vì giai đoạn kiểm tra chậm hơn hẳn giai đoạn học: ticket ở tập học mất trung vị "
                f"**{vn(train_days, 1)} ngày** đến Resolve, ở tập kiểm tra **{vn(test_days, 1)} ngày** (từ quý 4/2012 mọi ticket chậm hơn, màn hình 2).")

    st.subheader("Không đoán trước được ticket nào sẽ phải chờ (Wait)")
    cols = st.columns(len(gate_guesses))
    for col, (model, guesses) in zip(cols, gate_guesses.items()):
        col.metric(model, f"{guesses} lần", help="Số lần mô hình đoán bước tiếp theo là Wait")
    st.caption(f"Lúc ticket vừa được nhận (Take in charge ticket lần đầu): {vn(gate_n)} ticket kiểm tra, thực tế "
               f"{vn(round(gate_n * gate_wait))} ticket ({gate_wait:.0%}) có bước ngay sau là Wait. Số trên là số lần mỗi mô hình đoán \"sẽ chờ\".")

    st.subheader("Hẹn nhắc theo số ngày dự đoán không tốt hơn hẹn cố định")
    rem = reminders.loc[MODELS]
    st.dataframe(pd.DataFrame({
        "hẹn sau (ngày)": rem["hẹn (trung vị ngày)"],
        "bị nhắc": rem["lần Wait bị nhắc"],
        "bớt (ngày)": rem["bớt / lần Wait (ngày)"],
        "cố định: X (ngày)": rem["cố định cùng số lần nhắc: X (ngày)"],
        "cố định: bớt (ngày)": rem["cố định: bớt / lần Wait (ngày)"],
    }).style.format({"bị nhắc": "{:.0%}"}, precision=1), width="stretch")
    st.caption(f"{vn(reminders['waits'].iloc[0])} lần ticket kiểm tra vào Wait. Hẹn sau: trung vị số ngày mô hình đoán còn lại, "
               "dùng làm ngày hẹn nhắc. Bị nhắc: % lần Wait chờ quá ngày hẹn. Bớt: số ngày chờ cắt được trung bình mỗi lần Wait, "
               "giả định nhắc xong ticket đi tiếp ngay. Cố định: hẹn cùng X ngày cho mọi ticket, X chọn để có cùng số lần nhắc.")
    st.markdown("Cùng số lần nhắc, hẹn cố định bớt được nhiều bằng hoặc hơn. "
                "**Vì vậy dùng cách nhắc khi chờ quá X ngày ở màn hình 2.**")

    with st.expander("Cách làm"):
        st.markdown(f"""
- **Dữ liệu:** xếp ticket theo ngày mở; ticket xong trước {cut:%d/%m/%Y} để học, ticket mở từ ngày đó để kiểm tra.
- **Vector bước:** Node2Vec trên BPMN thực tế, mỗi loại bước một vector.
- **Thời gian còn lại:** số ngày đến lần Resolve ticket đầu tiên; không tính khoảng chờ đóng ticket sau đó.
- **Baseline:** bước hay đi sau bước hiện tại nhất; thời gian còn lại là trung vị theo bước hiện tại.
- **Gradient Boosting:** gộp ticket thành một dòng số, không biết thứ tự bước.
- **LSTM:** đọc lần lượt từng bước.
- **Transformer:** nhìn mọi bước cùng lúc (attention).
- **Decision Tree:** tại điểm rẽ sau Take in charge ticket, học từ người nhận, thứ, giờ, số bước trước và thời gian chờ đến lúc nhận.
""")
