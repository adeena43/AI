# Grid representation
grid = [
    ['O', 'O', 'X', 'O', 'T'],
    ['O', 'X', 'O', 'O', 'X'],
    ['P', 'O', 'O', 'X', 'O'],
    ['X', 'X', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'X', 'O']
]

# Possible moves (Up, Down, Left, Right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Find start and target positions
def find_positions(grid):
    start, target = None, None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 'P':
                start = (i, j)
            elif grid[i][j] == 'T':
                target = (i, j)
    return start, target

start, target = find_positions(grid)

# Convert Grid to Graph
def create_graph(grid):
    rows = len(grid)
    cols = len(grid[0])
    graph = {}

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 'X':  # Ignore obstacles
                neighbors = []
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != 'X':
                        neighbors.append((nx, ny))
                graph[(i, j)] = neighbors

    return graph

# BFS Algorithm
def bfs(graph, start, goal):
    visited = []
    queue = [start]
    visited.append(start)

    while queue:
        node = queue.pop(0)  # FIFO for BFS
        print(node, end=" ")  # Print visited node

        if node == goal:
            print("\nGoal reached")
            return

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)

    print("\nGoal not found")

# DFS Algorithm
def dfs(graph, start, goal):
    visited = []
    stack = [start]
    visited.append(start)

    while stack:
        node = stack.pop()  # LIFO for DFS
        print(node, end=" ")  # Print visited node

        if node == goal:
            print("\nGoal reached")
            return

        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                visited.append(neighbor)
                stack.append(neighbor)

    print("\nGoal not found")

# Convert grid to graph
graph = create_graph(grid)

# Run BFS
print("\nSearching for goal using BFS:\n")
bfs(graph, start, target)

# Run DFS
print("\nSearching for goal using DFS:\n")
dfs(graph, start, target)
