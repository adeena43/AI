import random

def initialize_system():
    components = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    system_state = {component: random.choice(["Safe", "Vulnerable"]) for component in components}
    return system_state

def scan_system(system_state):
    vulnerable_components = []
    print("\nSystem Scan Results:")
    for component, state in system_state.items():
        if state == "Vulnerable":
            print(f"WARNING: Component {component} is vulnerable.")
            vulnerable_components.append(component)
        else:
            print(f"SUCCESS: Component {component} is safe.")
    return vulnerable_components

def patch_vulnerabilities(system_state, vulnerable_components):
    print("\nPatching Vulnerabilities:")
    for component in vulnerable_components:
        system_state[component] = "Safe"
        print(f"Component {component} has been patched and is now safe.")
    return system_state

def display_system_state(system_state, title):
    print(f"\n{title}")
    for component, state in system_state.items():
        print(f"Component {component}: {state}")

def main():
    print("Cybersecurity System Simulation\n" + "=" * 35)

    system_state = initialize_system()
    display_system_state(system_state, "Initial System State")

    vulnerable_components = scan_system(system_state)

    if vulnerable_components:
        system_state = patch_vulnerabilities(system_state, vulnerable_components)
    else:
        print("\nNo vulnerabilities detected. No patching required.")

    display_system_state(system_state, "Final System State")

if __name__ == "__main__":
    main()

