from queue import PriorityQueue
from operator import itemgetter
import heapq

graph = {
    'S': [('A', 3), ('B', 6), ('C', 5)],
    'A': [('D', 9), ('E', 8)],
    'B': [('F', 12), ('G', 14)],
    'C': [('H', 7)],
    'H': [('I', 5), ('J', 6)],
    'I': [('K', 1), ('L', 10), ('M', 2)],
    'D': [], 'E': [], 'F': [], 'G': [], 'J': [],
    'K': [], 'L': [], 'M': []  # Leaf nodes
}

def beamSearch(start, target, beam_width=2):
    beam = [(0, [start])]

    while beam:
        candidates = []
        for cost, path in beam:
            current_node = path[-1]
            if current_node == target:
                return path, cost
            
            for neighbor, edge_cost in graph.get(current_node):
                new_cost = cost+edge_cost
                new_path = path + [neighbor]
                candidates.append((new_cost, new_path))

            beam = heapq.nsmallest(beam_width, candidates, key=itemgetter(0))

    return None, float('inf')


startNode = 'S'
target = 'L'
beam_width = 3

path, cost = beamSearch(start=startNode, target=target, beam_width=beam_width)
if path:
    print(f"Path found: {' → '.join(path)} with total cost: {cost}")
else:
    print("No path found.") 
