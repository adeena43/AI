class Environment:
    def __init__(self, initial_state = 0):
        self.initial_state = initial_state
    def get_percept(self):
        return self.initial_state

class SimpleReflexAgent:
    def __init__(self):
        pass
    def act(self, percept):
        if percept >= 100:
            print(f"The the total cost is: {10*percept}")
            return 10*percept
        else:
            print(f"the total cost is: {15*percept}")
            return 15*percept

def run_agent(agent, environment):
    # The agent reacts to the initial stimulus/Percept
    percept = environment.get_percept()
    action = agent.act(percept)
    print(f"Percept: {percept}, Action: {action}")

# Create instances of agent and environment
agent = SimpleReflexAgent()

environment = Environment(initial_state=150) 
run_agent(agent, environment)
