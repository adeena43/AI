tree = {
    'C': ['myFile', 'Ai_data'],
    'myFile': ['CS_PP'],
    'Ai_data': [],
    'CS_PP': ['Fall24', 'VS_extension']
}

def bfs(tree, start, goal):
    visited = [] 
    queue = [] 
    visited.append(start)
    queue.append(start)
    while queue:
        node = queue.pop(0) 
        print(node, end=" ")
        if node == goal: 
            print("\nGoal found!")
            break
        for neighbour in tree[node]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

start_node = 'C'
goal_node = "Fall24"

print("Running the bfs algorithm to find the CS past papers in this PC")

bfs(tree, start_node, goal_node)
