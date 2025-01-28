class Environment:
    def __init__(self):
        self.list = ['Failed', 'Failed', 'Completed', 'Completed', 'Failed']

    def display(self):
        print(f"List: {self.list}")

class AgentClass:
    def __init__(self):
        pass
    
    def scanSystem(self, list):
        for i in range(len(list)):
            if(list[i] == "Failed"):
                print("Retrying...")
                list[i] = "Completed"
                print("Process completed successfully!")

            else:
                print("Process already completed.")

if __name__ == "__main__":
    env = Environment()
    agent = AgentClass()
    print("Current status of processes: ")
    env.display()
    print("Status after retrying: ")
    agent.scanSystem(env.list)
    env.display()
