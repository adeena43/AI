from queue import PriorityQueue

graph = {
    'A': [('B', 5, 9), ('C', 8, 5)], # (neighbor, cost, heuristic)
    'B': [('D', 10, 4)], # (neighbor, cost, heuristic)
    'C': [('E', 3, 7)], # (neighbor, cost, heuristic)
    'D': [('F', 7, 5)], # (neighbor, cost, heuristic)
    'E': [('F', 2, 1)], # (neighbor, cost, heuristic)
    'F': [] # (neighbor, cost, heuristic)
}

def a_star(graph, start, goal):
    visited = set()
    pq = PriorityQueue()

    pq.put((0, start))
    while not pq.empty():
        cost, node = pq.get()
        if node not in visited:
            print(node, end= " ")
            visited.add(node)
            if node == goal:
                print("\nGoal reached")
                return True
            for neighbor, edge_cost, heuristic in graph[node]:
                if neighbor not in visited:
                    f_value = cost + edge_cost + heuristic
                    pq.put((f_value, neighbor))

    print("\nGoal not reachable")
    return False

print("\nA star path: ")
a_star(graph, 'A', 'F')
