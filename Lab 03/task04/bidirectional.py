from collections import deque

class BidirectionalSearch:
    def __init__(self, graph):
        self.graph = graph

    def bidirectionalSearch(self, start, goal):
        if start not in self.graph or goal not in self.graph:
            return None
        start_queue = deque([start])
        goal_queue = deque([goal])

        start_visited = {start: None}
        goal_visited = {goal:None}

        while start_queue and goal_queue:
            if self.bfsStep(start_queue, start_visited, goal_visited):
                return self.contructPath(start_visited, goal_visited)
            
            if self.bfsStep(goal_queue, goal_visited, start_visited):
                return self.contructPath(start_visited, goal_visited)
            
        return None
    
    def bfsStep(self, queue, visited, visitedOther):
        node = queue.popleft()
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                visited[neighbor]=node
                queue.append(neighbor)
                if neighbor in visitedOther:
                    return True
        return False
    
    def contructPath(self, start_visited, goal_visited):
        intersection = set(start_visited.keys()) & set(goal_visited.keys())
        if not intersection:
            return None
        meetingPoint = intersection.pop()

        path = []
        node = meetingPoint
        while node is not None:
            path.append(node)
            node = start_visited[node]

        path.reverse()

        node = goal_visited[meetingPoint]
        while node is not None:
            path.append(node)
            node = goal_visited[node]

        return path
    
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'F'],
    'E': ['C', 'F'],
    'F': ['D', 'E', 'G'],
    'G': ['F', 'H'],
    'H': ['G', 'I'],
    'I': ['H', 'J'],
    'J': ['I']
}

search = BidirectionalSearch(graph)
path = search.bidirectionalSearch('A', 'J')
print("Shortest path:", path)
