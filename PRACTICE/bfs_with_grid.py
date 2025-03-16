maze = [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1]
]

directions = [(0, 1), (1, 0)]

def create_graph(maze):
    graph = {}
    rows = len(maze)
    cols = len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                neighbors = []
                for dx, dy in directions:
                    nx, ny = i +dx, j+ dy
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                        neighbors.append((nx, ny))
                graph[(i, j)] = neighbors

    return graph
graph = create_graph(maze)
print(graph)

def bfs(graph, start, goal):
    visited = []
    queue = []

    visited.append(start)
    queue.append(start)

    while queue:
        node = queue.pop()
        print(node, end = " ")
        if node == goal:
            print("\nGoal reached")
        
        for neighbors in graph.get(node, []):
            if neighbors not in visited:
                visited.append(neighbors)
                queue.append(neighbors)

snode = (0, 0)
gnode = (2, 2)

print("\nFollowing is the Breadth-First Search (BFS): ")
bfs(graph, snode, gnode)
