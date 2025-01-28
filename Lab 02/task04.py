class SystemEnvironment:
    def __init__(self):
        self.components = ['Safe', 'Low', 'Vulnerable', 'High', 'High', 'Low', 'Vulnerable', 'High']

    def display_status(self):
        print("Current System Vulnerabilities:")
        for component in self.components:
            print(f"{component} ")
        print("\n")

class SecurityAgent:
    def __init__(self):
        self.patched = []

    def scan_system(self, components):
        print("System Scan Report:")
        for component in components:
            if component in ["Vulnerable", "Low", "High"]:
                print("Warning: System is Vulnerable!\n")
            else:
                print("System is Secure! No vulnerabilities found.\n")

    def patch_vulnerabilities(self, components):
        print("Patching Process:")
        for i, component in enumerate(components):
            if component == "Low":
                self.patched.append(i)
                print(f"Low risk vulnerability in component {i} patched successfully!\n")
            elif component == "High":
                print(f"Component {i} has a high-risk vulnerability. Premium service required.\n")

    def final_check(self, components):
        low_risk_count = components.count("Low")
        patched_count = len(self.patched)

        if patched_count == low_risk_count:
            print("All Low Risk Vulnerabilities have been successfully patched!")
        else:
            print("Some Low Risk Vulnerabilities remain unpatched.")

def main():
    system = SystemEnvironment()
    agent = SecurityAgent()
    system.display_status()
    agent.scan_system(system.components)
    agent.patch_vulnerabilities(system.components)
    agent.final_check(system.components)

if __name__ == "__main__":
    main()
