"""BPMN của quy trình: máy dựng từ log và quy trình chuẩn do nhóm vẽ; so từng ticket với quy trình chuẩn."""
import networkx as nx
import pandas as pd
import pm4py
from pm4py.objects.bpmn.obj import BPMN
from pm4py.util import constants

from helpdesk_app.event_log import ACTIVITY, CASE, TIMESTAMP, traces

constants.SHOW_PROGRESS_BAR = False

# Giả định của nhóm: đường đi của một ticket bình thường.
STANDARD_PROCESS = ["Assign seriousness", "Take in charge ticket", "Resolve ticket", "Closed"]


def _pm4py_log(event_log: pd.DataFrame) -> pd.DataFrame:
    return pm4py.format_dataframe(event_log, case_id=CASE, activity_key=ACTIVITY, timestamp_key=TIMESTAMP)


def standard_bpmn() -> BPMN:
    steps = ", ".join(f"'{activity}'" for activity in STANDARD_PROCESS)
    return pm4py.convert_to_bpmn(pm4py.parse_process_tree(f"->({steps})"))


def discover_bpmn(event_log: pd.DataFrame, top_variants: int = 5) -> tuple[BPMN, float]:
    """Inductive Miner trên các ticket thuộc `top_variants` đường đi phổ biến nhất; trả BPMN và tỷ lệ ticket được dùng.

    Dùng cả log thì BPMN có hàng chục gateway, không đọc được; 5 đường đi đầu đã phủ khoảng 3/4 số ticket.
    """
    case_traces = traces(event_log)
    common = set(case_traces.value_counts().index[:top_variants])
    kept = case_traces[case_traces.isin(common)].index
    bpmn = pm4py.discover_bpmn_inductive(_pm4py_log(event_log[event_log[CASE].isin(kept)]))
    return bpmn, len(kept) / len(case_traces)


def compare_with_standard(event_log: pd.DataFrame) -> pd.DataFrame:
    """Chuyển BPMN chuẩn sang Petri Net rồi căn từng ticket theo mô hình (alignment).

    Chọn alignment thay vì token replay: token replay của pm4py bỏ qua activity không có trong mô hình (như Wait),
    nên ticket có Wait vẫn bị tính là khớp. Mỗi dòng: ticket có khớp chuẩn không, các bước thừa/thiếu, và từng bước đã đối chiếu.
    """
    net, initial, final = pm4py.convert_to_petri_net(standard_bpmn())
    log = pm4py.convert_to_event_log(_pm4py_log(event_log))
    results = pm4py.conformance_diagnostics_alignments(log, net, initial, final)
    rows = []
    for trace, result in zip(log, results):
        # Mỗi bước của ticket: (tên bước, "khớp" / "thừa" / "thiếu"); bỏ bước ẩn của Petri Net.
        steps = [(log_step, "thừa") if model_step == ">>" else (model_step, "thiếu") if log_step == ">>" else (log_step, "khớp")
                 for log_step, model_step in result["alignment"] if not (log_step == ">>" and model_step is None)]
        deviations = [f"{kind} {step}" for step, kind in steps if kind != "khớp"]
        rows.append({CASE: trace.attributes["concept:name"], "fits": not deviations, "deviations": deviations, "steps": steps})
    return pd.DataFrame(rows)


def bpmn_graph(bpmn: BPMN) -> nx.DiGraph:
    """BPMN thành đồ thị có hướng; mỗi node có `label` (chữ hiển thị) và `kind` (Start/End/Task/XOR/AND)."""
    kinds = {"StartEvent": "Start", "NormalStartEvent": "Start", "EndEvent": "End", "NormalEndEvent": "End",
             "Task": "Task", "ExclusiveGateway": "XOR", "ParallelGateway": "AND"}
    graph = nx.DiGraph()
    counter: dict[str, int] = {}
    names = {}
    for node in bpmn.get_nodes():
        kind = kinds.get(type(node).__name__, type(node).__name__)
        if kind == "Task":
            name = node.get_name()
        else:
            counter[kind] = counter.get(kind, 0) + 1
            name = f"{kind} {counter[kind]}" if kind in ("XOR", "AND") else kind
        names[node] = name
        graph.add_node(name, kind=kind, label=node.get_name() if kind == "Task" else {"XOR": "X", "AND": "+"}.get(kind, ""))
    for flow in bpmn.get_flows():
        graph.add_edge(names[flow.get_source()], names[flow.get_target()])
    return graph


REMINDER_STEP = "Nhắc tự động khi quá hạn chờ"


def to_be_graph() -> nx.DiGraph:
    """BPMN sau cải tiến (vẽ tay): giữ nhánh Wait; sau Wait, XOR 3 rẽ sang bước nhắc chỉ khi ticket chờ quá X ngày."""
    graph = nx.DiGraph()
    nodes = [("Start", "Start", ""), ("Assign seriousness", "Task", "Assign seriousness"),
             ("Take in charge ticket", "Task", "Take in charge ticket"), ("XOR 1", "XOR", "X"), ("Wait", "Task", "Wait"),
             ("XOR 3", "XOR", "X"), (REMINDER_STEP, "Task", REMINDER_STEP), ("XOR 2", "XOR", "X"), ("Resolve ticket", "Task", "Resolve ticket"),
             ("Closed", "Task", "Closed"), ("End", "End", "")]
    for name, kind, label in nodes:
        graph.add_node(name, kind=kind, label=label)
    graph.add_edges_from([("Start", "Assign seriousness"), ("Assign seriousness", "Take in charge ticket"),
                          ("Take in charge ticket", "XOR 1"), ("XOR 1", "Wait"), ("Wait", "XOR 3"),
                          ("XOR 3", REMINDER_STEP), ("XOR 3", "XOR 2"), (REMINDER_STEP, "XOR 2"), ("XOR 1", "XOR 2"), ("XOR 2", "Resolve ticket"),
                          ("Resolve ticket", "Closed"), ("Closed", "End")])
    return graph


def graph_dot(graph: nx.DiGraph, colors: dict[str, str] | None = None) -> str:
    """DOT để Streamlit vẽ; `colors` tô nền node (dùng cho kết quả gom nhóm)."""
    shapes = {"Start": "circle", "End": "doublecircle", "Task": "box", "XOR": "diamond", "AND": "diamond"}
    # Thứ tự đi từ Start; cạnh quay ngược (vòng làm lại) không được kéo bố cục, nên sơ đồ luôn chạy trái sang phải.
    start = next(n for n, kind in graph.nodes(data="kind") if kind == "Start")
    order = {n: i for i, n in enumerate(nx.bfs_tree(graph, start))}
    lines = ["digraph { rankdir=LR; node [fontname=Helvetica, fontsize=14];"]
    for name in sorted(graph.nodes, key=lambda n: order.get(n, len(order))):
        data = graph.nodes[name]
        fill = (colors or {}).get(name, "white")
        lines.append(f'"{name}" [label="{data["label"]}", shape={shapes[data["kind"]]}, style="{'rounded,filled' if data['kind'] == 'Task' else 'filled'}", fillcolor="{fill}"];')
    for a, b in sorted(graph.edges, key=lambda e: order.get(e[0], len(order))):
        back = order.get(b, 0) <= order.get(a, 0)
        lines.append(f'"{a}" -> "{b}"' + (" [constraint=false];" if back else ";"))
    return "\n".join(lines + ["}"])
