class Environment:
    def __init__(self):
        self.servers = ["Overload", "Underload", "Balance", "Overload", "Underload"]
    
    def display(self):
        print(f"System's current load status: {self.servers}")

class Agent:
    def __init__(self):
        pass
    
    def scanSystem(self, servers):
        overloaded = []
        underloaded = []

        for i in range(len(servers)):
            if (servers[i] == "Overload"):
                overloaded.append(i)
            elif (servers[i] == "Underload"):
                underloaded.append(i)
            else:
                pass
        
        while (len(overloaded)>0 and len(underloaded)>0):
            overloaded_index = overloaded.pop(0)
            underloaded_index = underloaded.pop(0)

            servers[overloaded_index] = "Balanced"
            servers[underloaded_index] = "Balanced"
    
    def display(self, servers):
        print(f"System's load status after updating: {servers}")

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    print("Initial server's load status: ")
    env.display()

    agent.scanSystem(env.servers)

    print("Final server's load status: ")
    agent.display(env.servers)
