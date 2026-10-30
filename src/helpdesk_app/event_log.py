"""Đọc event log Helpdesk; giữ nguyên tên cột và tên activity như file gốc."""
from pathlib import Path

import pandas as pd

CASE = "case:concept:name"
ACTIVITY = "Activity"
RESOURCE = "Resource"
TIMESTAMP = "time:timestamp"

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "dataset_Helpdesk_eventlog.csv"


def load_event_log(path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Sắp mỗi ticket theo thời gian; cùng timestamp thì giữ thứ tự dòng trong file để kết quả lặp lại được."""
    event_log = pd.read_csv(path, parse_dates=[TIMESTAMP])
    event_log["source_row"] = range(len(event_log))
    return event_log.sort_values([CASE, TIMESTAMP, "source_row"]).reset_index(drop=True)


def traces(event_log: pd.DataFrame) -> pd.Series:
    """Chuỗi activity của từng ticket."""
    return event_log.groupby(CASE)[ACTIVITY].agg(tuple)
