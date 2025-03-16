maze = [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1]
]

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def convert_grid(maze):
    rows = len(maze)
    cols = len(maze[0])
    graph = {}

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                neighbors = []
                for dx, dy in directions:
                    nx, ny = dx + i, dy + j
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                        neighbors.append((nx, ny))

                graph[(i, j)] = neighbors

    return graph

def dfs(graph, start, goal):
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue  # Skip if already visited

        print(node, end=" ")  # Print when visiting
        visited.add(node)

        if node == goal:
            print("\nGoal reached!")
            return  # Exit once goal is found

        # Push unvisited neighbors onto stack (in reverse order for correct DFS)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    print("\nGoal not found")
    return 

graph = convert_grid(maze)
dfs(graph, (0, 0), (2, 2))
