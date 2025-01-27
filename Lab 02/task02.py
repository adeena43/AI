import random

class Server:
    def __init__(self, name):
        self.name = name
        self.load_state = random.choice(['Underloaded', 'Balanced', 'Overloaded'])

    def __str__(self):
        return f"{self.name}: {self.load_state}"

class DataCenter:
    def __init__(self, num_servers=5):
        self.servers = [Server(f"Server {i + 1}") for i in range(num_servers)]
    
    def display_load_status(self):
        print("Current Load Status of Servers:")
        for server in self.servers:
            print(server)

class LoadBalancerAgent:
    def __init__(self):
        pass

    def balance_load(self, data_center):
        overloaded_servers = [server for server in data_center.servers if server.load_state == 'Overloaded']
        underloaded_servers = [server for server in data_center.servers if server.load_state == 'Underloaded']

        for overloaded in overloaded_servers:
            if underloaded_servers:
                print(f"Moving tasks from {overloaded.name} to {underloaded_servers[0].name}")
                overloaded.load_state = 'Balanced'
                underloaded_servers[0].load_state = 'Balanced'
                underloaded_servers = underloaded_servers[1:]

def run_load_balancer_simulation():
    data_center = DataCenter()
    data_center.display_load_status()
    load_balancer = LoadBalancerAgent()
    load_balancer.balance_load(data_center)
    print("\nUpdated Load Status After Balancing:")
    data_center.display_load_status()

run_load_balancer_simulation()
