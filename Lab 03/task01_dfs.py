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
    
    def dfs_search(self, start, goal):
        stack = [start]  
        visited = [] 

        while stack:
            node = stack.pop()  
            print(f"Visiting: {node}") 

            if node == goal: 
                print(f"Goal {goal} found!")
                return True  

            if node not in visited: 
                visited.append(node)  
               
                for neighbor in tree.get(node, []): 
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        print("Goal not found.")
        return False  
    
class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal reached"
        return "Searching"
    
    def act(self, percept, environment):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found"
        else:
            return environment.dfs_search(percept, self.goal)
        
def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment)
    print(action)

start_node = 'A'
goal_node = 'D'

agent =GoalBasedAgent(goal_node)
environment = Environment(tree)

run_agent(agent, environment, start_node)
