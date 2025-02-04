tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['B', 'C'],
    'E': ['H'],
    'F': ['I'],
    'G': [],
    'H': [],
    'I': []
}

class Environment:
    def __init__(self, graph):
        self.graph = graph

    def get_percept(self, node):
        return node

    def depth_limited_search_recursive(self, tree, current_node, goal, limit, visited=None):
        if visited is None:
            visited = []  
        print(f"Visiting: {current_node}, Depth: {limit}")  

        if current_node == goal:
            print(f"Goal {goal} found!")
            return True

        if limit == 0:
            return False
        visited.append(current_node)

        for neighbor in tree.get(current_node, []):
            if neighbor not in visited:
                if self.depth_limited_search_recursive(tree, neighbor, goal, limit - 1, visited):
                    return True

        return False

class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal

    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal reached"
        return "Searching"

    def act(self, percept, environment, start_node, goal_node, depth_limit=3):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found"
        else:
            
            found = environment.depth_limited_search_recursive(environment.graph, start_node, goal_node, depth_limit)
            if found:
                return f"Goal {goal_node} found at depth {depth_limit}"
            else:
                return f"Goal {goal_node} not found at depth {depth_limit}"

def run_agent(agent, environment, start_node, goal_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment, start_node, goal_node)
    print(action)

start_node = 'A'
goal_node = 'D'

agent = GoalBasedAgent(goal_node)
environment = Environment(tree)

run_agent(agent, environment, start_node, goal_node)
