
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'G'],
    'D': ['B'],
    'E': ['B'],
    'F': ['C'],
    'G': ['C']
}
def dls(node, goal, depth, path, visited):
    if depth == 0:
        return False
    if node == goal:
        path.append(node)
        return True
    visited.add(node)
    if node not in graph:
        return False
    for child in graph[node]:
        if child not in visited:
            if dls(child, goal, depth - 1, path, visited):
                path.append(node) 
                return True
    return False

def iterative_deepening(start, goal, max_depth):
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}")
        visited = set()  # Keep track of visited nodes to avoid cycles
        path = []
        if dls(start, goal, depth, path, visited):
            print("\nPath to goal:", " → ".join(reversed(path))) 
            return
    print("Goal not found within depth limit.")

start_node = 'A'
goal_node = 'G'
max_search_depth = 5
iterative_deepening(start_node, goal_node, max_search_depth)
