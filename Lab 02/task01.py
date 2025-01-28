import random

class Evironment:
    def __init__(self):
        self.criticalComponent = [random.choice([0,1]) for _ in range(9)]
        print(f"System's initial state: {self.criticalComponent} \n")

class Agent:
    def __init__(self):
        self.patch = []

    def systemScan(self, criticalComponent):
        for i in range(9):
            if (criticalComponent[i] == 0):
                print("Warning \n")
                self.patch.append(i)
            else:
                print("Logged \n")

    def patchingVulnerabilities(self, criticalComponent):
        for i in range(len(self.patch)):
            criticalComponent[self.patch[i]] = 1

    def finalSystemCheck(self, criticalComponent):
        print(f"Final state of the system: {criticalComponent}")

environent = Evironment()
agent1 = Agent()

agent1.systemScan(environent.criticalComponent)
agent1.patchingVulnerabilities(environent.criticalComponent)
agent1.finalSystemCheck(environent.criticalComponent)
