"""Buổi 6: biến từng node của BPMN thành vector (Node2Vec) rồi gom nhóm (K-Means); so cả BPMN với nhau (Graph2Vec)."""
import random

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans


def node2vec(graph: nx.DiGraph, dim: int = 8, walks_per_node: int = 100, walk_length: int = 8,
             p: float = 1.0, q: float = 1.0, seed: int = 42) -> dict[str, np.ndarray]:
    """Random walk có thiên hướng p/q, rồi Word2Vec skip-gram trên các walk: node hay đi cạnh nhau có vector gần nhau.

    Walk đi trên bản vô hướng của BPMN để không bị kẹt ở End; hướng thực thi của BPMN không đổi.
    """
    rng = random.Random(seed)
    undirected = graph.to_undirected()

    def next_node(previous: str | None, current: str) -> str:
        options = sorted(undirected.neighbors(current))
        if previous is None:
            return rng.choice(options)
        weights = [1 / p if n == previous else 1.0 if undirected.has_edge(n, previous) else 1 / q for n in options]
        return rng.choices(options, weights=weights)[0]

    walks = []
    for start in sorted(undirected.nodes):
        for _ in range(walks_per_node):
            walk, previous = [start], None
            while len(walk) < walk_length:
                previous, current = (walk[-2] if len(walk) > 1 else None), walk[-1]
                walk.append(next_node(previous, current))
            walks.append(walk)
    model = Word2Vec(walks, vector_size=dim, window=5, min_count=1, sg=1, epochs=20, seed=seed, workers=1)
    return {node: model.wv[node] for node in graph.nodes}


def cluster_nodes(vectors: dict[str, np.ndarray], k: int = 3, seed: int = 42) -> pd.Series:
    """Nhóm của từng node. Số thứ tự nhóm không có nghĩa, phải nhìn các node trong nhóm để đọc."""
    names = sorted(vectors)
    labels = KMeans(k, random_state=seed, n_init=10).fit_predict(np.array([vectors[n] for n in names]))
    return pd.Series(labels, index=names, name="cluster")


def _wl_words(graph: nx.DiGraph, iterations: int = 2) -> list[str]:
    """Weisfeiler–Lehman: đổi tên từng node theo các node trước và sau nó; mỗi tên là một "từ" của đồ thị."""
    label = {n: (d["label"] if d["kind"] == "Task" else d["kind"]) for n, d in graph.nodes(data=True)}
    words = list(label.values())
    for _ in range(iterations):
        label = {n: f"{label[n]}|in:{','.join(sorted(label[p] for p in graph.predecessors(n)))}"
                    f"|out:{','.join(sorted(label[s] for s in graph.successors(n)))}" for n in graph.nodes}
        words += label.values()
    return words


def graph2vec_similarity(graphs: dict, seeds: int = 5) -> tuple[pd.DataFrame, float]:
    """Graph2Vec: mỗi BPMN là một văn bản gồm các từ WL → Doc2Vec → cosine giữa từng cặp.
    Trả về (trung bình cosine qua các seed, độ lệch chuẩn lớn nhất giữa các seed)."""
    sims = []
    for seed in range(seeds):
        model = Doc2Vec([TaggedDocument(_wl_words(g), [str(k)]) for k, g in graphs.items()],
                        vector_size=16, dm=0, min_count=1, epochs=200, seed=seed, workers=1)
        V = np.array([model.dv[str(k)] for k in graphs])
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        sims.append(V @ V.T)
    return pd.DataFrame(np.mean(sims, 0), index=list(graphs), columns=list(graphs)), float(np.std(sims, 0).max())


if __name__ == "__main__":
    from sklearn.metrics import silhouette_score

    from helpdesk_app.event_log import load_event_log
    from helpdesk_app.process_model import bpmn_graph, discover_bpmn

    graph = bpmn_graph(discover_bpmn(load_event_log())[0])
    for seed in range(5):
        vectors = node2vec(graph, seed=seed)
        clusters = cluster_nodes(vectors, seed=seed)
        score = silhouette_score(np.array([vectors[n] for n in clusters.index]), clusters)
        groups = clusters.groupby(clusters).apply(lambda s: sorted(s.index)).tolist()
        print(f"seed {seed} silhouette {score:.2f}: {groups}")
