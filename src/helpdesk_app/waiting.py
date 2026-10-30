"""Chỗ kẹt ở Wait: chờ bao lâu, tăng theo năm ra sao, ticket nào đang chờ, và mô phỏng nhắc khi chờ quá X ngày.

Log chỉ có thời điểm làm xong, nên "thời gian chờ" = khoảng cách từ lúc ghi Wait đến bước kế tiếp.
"""
import pandas as pd

from helpdesk_app.event_log import ACTIVITY, CASE, TIMESTAMP

DAY = pd.Timedelta(days=1)


def _with_next_gap(event_log: pd.DataFrame) -> pd.DataFrame:
    gap = (event_log.groupby(CASE)[TIMESTAMP].shift(-1) - event_log[TIMESTAMP]) / DAY
    return event_log.assign(gap_days=gap)


def wait_durations(event_log: pd.DataFrame) -> pd.Series:
    """Số ngày từ mỗi lần Wait đến bước kế tiếp (bỏ Wait nằm cuối ticket)."""
    events = _with_next_gap(event_log)
    return events.loc[events[ACTIVITY] == "Wait", "gap_days"].dropna().rename("wait_days")


def wait_share_by_year(event_log: pd.DataFrame) -> pd.DataFrame:
    """Theo năm mở ticket: số ticket và tỷ lệ ticket có Wait."""
    per_case = event_log.groupby(CASE).agg(year=(TIMESTAMP, lambda t: t.min().year),
                                           has_wait=(ACTIVITY, lambda a: (a == "Wait").any()))
    return per_case.groupby("year").agg(tickets=("has_wait", "size"), wait_share=("has_wait", "mean"))


def days_to_resolve(event_log: pd.DataFrame, remind_after: float | None = None) -> pd.DataFrame:
    """Mỗi ticket: số ngày từ bước đầu đến lần Resolve ticket đầu tiên.

    `remind_after`: mô phỏng TO-BE, ticket nằm ở Wait quá X ngày thì được nhắc và đi tiếp ngay sang bước kế tiếp
    (lần chờ dài hơn X bị cắt còn X); các khoảng khác giữ nguyên. Không tính Closed vì đó là thủ tục đóng sau khi
    đã giải quyết. Bỏ ticket không có Resolve ticket.
    """
    events = _with_next_gap(event_log)
    if remind_after is not None:
        events["gap_days"] = events["gap_days"].where(events[ACTIVITY] != "Wait", events["gap_days"].clip(upper=remind_after))
    is_resolve = events[ACTIVITY] == "Resolve ticket"
    before_resolve = events[~(is_resolve.groupby(events[CASE]).cumsum() > 0)]
    per_case = before_resolve.groupby(CASE).agg(days=("gap_days", "sum"),
                                                has_wait=(ACTIVITY, lambda a: (a == "Wait").any()))
    return per_case[per_case.index.isin(events.loc[is_resolve, CASE].unique())]


def by_quarter(event_log: pd.DataFrame) -> pd.DataFrame:
    """Theo quý mở ticket: số ticket, tỷ lệ có Wait, trung vị số ngày đến Resolve ticket (có / không có Wait)
    và trung vị số ngày chờ người nhận (Assign seriousness → Take in charge ticket) và chờ đóng (Resolve ticket → Closed)."""
    quarter = event_log.groupby(CASE)[TIMESTAMP].min().dt.tz_localize(None).dt.to_period("Q").astype(str)
    resolve = days_to_resolve(event_log).join(quarter.rename("quarter"))
    events = _with_next_gap(event_log).assign(next=lambda e: e.groupby(CASE)[ACTIVITY].shift(-1))
    events["quarter"] = events[CASE].map(quarter)

    def gap(a: str, b: str) -> pd.Series:
        return events[(events[ACTIVITY] == a) & (events["next"] == b)].groupby("quarter")["gap_days"].median()

    return pd.DataFrame({
        "tickets": resolve.groupby("quarter").size(),
        "wait_share": resolve.groupby("quarter")["has_wait"].mean(),
        "days_with_wait": resolve[resolve["has_wait"]].groupby("quarter")["days"].median(),
        "days_without_wait": resolve[~resolve["has_wait"]].groupby("quarter")["days"].median(),
        "days_to_take": gap("Assign seriousness", "Take in charge ticket"),
        "days_to_close": gap("Resolve ticket", "Closed"),
    })


def waiting_at(event_log: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Ticket đang nằm ở Wait vào ngày `today`: bước gần nhất trước đó là Wait. Đã chờ = today − time:timestamp của Wait."""
    seen = event_log[event_log[TIMESTAMP] <= today]
    last = seen.groupby(CASE).tail(1)
    waiting = last[last[ACTIVITY] == "Wait"]
    return waiting.assign(days_waiting=(today - waiting[TIMESTAMP]) / DAY).sort_values("days_waiting", ascending=False)


if __name__ == "__main__":
    from helpdesk_app.event_log import load_event_log

    event_log = load_event_log()
    waits = wait_durations(event_log)
    print(f"Wait: {len(waits)} lần, trung vị {waits.median():.1f} ngày")
    for x in (None, 7):
        result = days_to_resolve(event_log, x)
        print(f"nhắc sau {x} ngày: ticket có Wait trung vị {result.loc[result['has_wait'], 'days'].median():.1f} ngày")
    today = pd.Timestamp("2013-06-01", tz="UTC")
    print(f"{today.date()}: {len(waiting_at(event_log, today))} ticket đang ở Wait")
