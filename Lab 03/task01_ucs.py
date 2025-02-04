import heapq

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2, 'E': 5},
    'C': {'A': 4, 'F': 3},
    'D': {'B': 2, 'G': 7},
    'E': {'B': 5, 'H': 6},
    'F': {'C': 3},
    'G': {'D': 7},
    'H': {'E': 6},
}

class Environment:
    def __init__(self, graph):
        self.graph = graph

    def get_percept(self, node):
        return node

    def uniform_cost_search(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start)) 
        cost_so_far = {start: 0}

        came_from = {start: None}

        while frontier:
            current_cost, current_node = heapq.heappop(frontier)
            if current_node == goal:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = came_from[current_node]
                return path[::-1], cost_so_far[goal] 
            for neighbor, cost in self.graph.get(current_node, {}).items():
                new_cost = current_cost + cost

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current_node
                    heapq.heappush(frontier, (new_cost, neighbor))

        return None, float('inf')

class UtilityBasedAgent:
    def __init__(self, goal):
        self.goal = goal

    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal reached"
        return "Searching"
    
    def act(self, percept, environment, start, goal):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found"
        else:
            path, cost = environment.uniform_cost_search(start, goal)
            if path:
                return f"Path to {goal} found: {path}, Total cost: {cost}"
            else:
                return f"Goal {goal} not found"

def run_agent(agent, environment, start_node, goal_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment, start_node, goal_node)
    print(action)

start_node = 'A'
goal_node = 'G'

agent = UtilityBasedAgent(goal_node)
environment = Environment(graph)

run_agent(agent, environment, start_node, goal_node)
