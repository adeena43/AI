import random

# Function to count conflicts (attacking queen pairs)
def calculate_conflicts(state):
    conflicts = 0
    n = len(state)

    for i in range(n):
        for j in range(i + 1, n):  
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1

    return conflicts

# Function to generate neighboring states
def get_neighbors(state):
    neighbors = []
    n = len(state)

    for row in range(n):
        for col in range(n):
            if col != state[row]:  # Change column of one queen
                new_state = list(state)
                new_state[row] = col
                neighbors.append(new_state)

    return neighbors

# Hill Climbing Algorithm
def hill_climb(n):
    # Generate a random initial state
    current_state = [random.randint(0, n - 1) for _ in range(n)]
    current_conflicts = calculate_conflicts(current_state)

    while True:
        neighbors = get_neighbors(current_state)
        next_conflicts = current_conflicts

        # Check for the best neighbor with fewer conflicts
        for neighbor in neighbors:
            neighbor_conflicts = calculate_conflicts(neighbor)
            if neighbor_conflicts < next_conflicts:
                next_state = neighbor
                next_conflicts = neighbor_conflicts

        # If no better neighbor is found, stop
        if next_conflicts >= current_conflicts:
            break

        # Move to the better neighbor
        current_state = next_state
        current_conflicts = next_conflicts

    return current_state, current_conflicts

# Run the Hill Climbing Algorithm for N-Queens
n = 8
solution, conflicts = hill_climb(n)

# Print results
if conflicts == 0:
    print(f"Solution found for {n}-Queens problem:")
    print(solution)
else:
    print(f"Could not find a solution. Stuck at state with {conflicts} conflicts:")
    print(solution)
