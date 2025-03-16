class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal

    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal reached"
        else:
            return "Searching"
        
    def act(self, percept, environment):
        goal_status = self.formulate_goal
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found"
        else:
            return environment.dfs_search(percept, self.goal)
        

class Environment:
    def __init__(self, graph):
        self.graph = graph

    def get_percept(self, node):
        return node
    
    def dfs_search(self, start, goal):
        visited = []
        stack = []

        visited.append(start)
        stack.append(start)

        while stack:
            node = stack.pop()
            print(f"Visiting: {node}")

            if node == goal:
                return f"Goal {goal} found"
            
            for neighbor in reversed(self.graph.get(node, [])):
                if neighbor not in visited:
                    visited.append(neighbor)
                    stack.append(neighbor)

        return "Goal not found"
    
def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment)
    print(action)

start_node = 'A'
goal_node = 'I'

agent = GoalBasedAgent(goal_node)
environment = Environment(tree)
run_agent(agent, environment, start_node)
