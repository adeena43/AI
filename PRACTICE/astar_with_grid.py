from queue import PriorityQueue

# Grid representation
maze = [
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 1]
]

# Possible movements (Right, Left, Down, Up)
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Manhattan distance
def heuristic(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return abs(x1-x2) + abs(y1-y2)

def create_graph(maze, goal):
    graph = {}
    rows = len(maze)
    cols = len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:  # If it's an open path
                neighbors = []
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                        edge_cost = 1  # Assume cost = 1 for each step
                        neighbors.append(( (nx, ny), edge_cost, heuristic((nx, ny), goal) ))
                graph[(i, j)] = neighbors  # Store neighbors with costs and heuristic
    return graph

def a_star(graph, start, goal):
    visited = set()
    pq = PriorityQueue()
    pq.put((0, start))

    parent = {start: None}

    while pq:
        cost, node = pq.get()
        if node not in visited:
            print(node, end = " ")
            visited.add(node)
            if node == goal:
                print("\nGoal reached")
                path = []
                while node:
                    path.append(node)
                    node = parent[node]

                return reversed(path)
            
            for neighbor, edge_cost, heuristic in graph.get(node, []):
                if neighbor not in visited:
                    f_value = edge_cost+cost+heuristic
                    pq.put((f_value, neighbor))
                    parent[neighbor] = node

    print("\nGoal not reachable")
    return None

start = (0, 0)
goal = (2, 2)
graph = create_graph(maze)

# Run A* search
print("\nFollowing is the A* Search Path:")
a_star_path = a_star(graph, start, goal)
print("\nA* Path:", a_star_path if a_star_path else "No path found")
