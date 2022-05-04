from typing import List
from itertools import chain
from collections import defaultdict


def kahn_sort(graph: dict, vertices: List[str] = None, revert=False):
    seen_vertices = list(set(graph.keys()) | set(chain.from_iterable(graph.values())))
    if vertices is None:
        vertices = seen_vertices
    else:
        msg = f"`vertices` must contain all vertices mentioned in the provided graph. `{set(seen_vertices) - set(vertices)}` not in provided vertices."
        assert set(seen_vertices).issubset(set(vertices)), msg

    vertice_dict = {vertice: idx for idx, vertice in enumerate(vertices)}
    n_vertices = len(vertices)

    int_graph = defaultdict(
        list,
        {
            vertice_dict[key]: [vertice_dict[_key] for _key in item]
            for key, item in graph.items()
        },
    )

    # vertices. Initialize all indegrees as 0.
    in_degree = [0 for _ in range(n_vertices)]
    for i in int_graph:
        for j in int_graph[i]:
            in_degree[j] += 1

    # Create an queue and enqueue all vertices with
    # indegree 0
    queue = []
    for i in range(n_vertices):
        if in_degree[i] == 0:
            queue.append(i)

    # Initialize count of visited vertices
    cnt = 0

    # Create a vector to store result (A topological
    # ordering of the vertices)
    top_order = []

    # One by one dequeue vertices from queue and enqueue
    # adjacents if indegree of adjacent becomes 0
    while queue:
        # Extract front of queue (or perform dequeue)
        # and add it to topological order
        u = queue.pop(0)
        top_order.append(u)

        # Iterate through all neighbouring nodes
        # of dequeued node u and decrease their in-degree
        # by 1
        for i in int_graph[u]:
            in_degree[i] -= 1
            # If in-degree becomes zero, add it to queue
            if in_degree[i] == 0:
                queue.append(i)

        cnt += 1

    # Check if there was a cycle
    if cnt != n_vertices:
        raise ValueError("There exists a cycle in the graph")
    if revert:
        top_order = top_order[::-1]
    return [vertices[idx] for idx in top_order]
