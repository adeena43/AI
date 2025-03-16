import heapq
from queue import PriorityQueue

# Grid representation
grid = [
    [1, 2, 3, '#', 4],
    [1, '#', 1, 2, 2],
    [2, 3, 1, '#', 1],
    ['#', '#', 2, 1, 1],
    [1, 1, 2, 2, 1]
]

# Possible moves (Up, Down, Left, Right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Manhattan Distance Heuristic
def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1 - x2) + abs(y1 - y2)

# Convert Grid to Graph
def create_graph(grid, goal):
    rows, cols = len(grid), len(grid[0])
    graph = {}

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != '#':  # Ignore obstacles
                neighbors = []
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != '#':
                        neighbors.append(((nx, ny), grid[nx][ny], heuristic((nx, ny), goal)))
                graph[(i, j)] = neighbors

    return graph

# A* Algorithm
def a_star(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    parent = {start: None}
    pq.put((0, start))  # (f-score, node)
    cost_so_far = {start: 0}

    while not pq.empty():
        cost, node = pq.get()  # Get the node with lowest f-score

        if node in visited:
            continue  # Skip if already processed

        visited.add(node)

        if node == goal:
            path = []
            print("\nGoal reached!")
            while node:
                path.append(node)
                node = parent[node]
            return list(reversed(path))  # Return reconstructed path
        
        for neighbor, edge_cost, heuristic_value in graph.get(node, []):
            new_cost = cost_so_far[node] + edge_cost  # g-score
            
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                f_score = new_cost + heuristic_value
                pq.put((f_score, neighbor))
                parent[neighbor] = node  # Track path

    print("\nGoal unreachable")
    return None

# Find start and goal positions
start = (0, 0)
goal = (len(grid) - 1, len(grid[0]) - 1)

# Convert grid to graph
graph = create_graph(grid, goal)

# Run A* search
print("\nSearching for optimal path using A*:\n")
path = a_star(graph, start, goal)

if path:
    print("Optimal Path:", path)
else:
    print("No valid path found.")
