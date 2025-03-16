import random

# Configuration
num_staff = 5  # Number of employees
num_shifts = 21  # 7 days * 3 shifts per day
max_shifts_per_staff = 7
required_staff_per_shift = 2
population_size = 10
mutation_rate = 0.1
max_generations = 100

# Function to evaluate fitness (lower is better)
def evaluate_fitness(schedule):
    penalty = 0

    # Check shift coverage
    for shift in range(num_shifts):
        assigned_count = 0
        for staff in range(num_staff):
            assigned_count += schedule[staff][shift]

        if assigned_count < required_staff_per_shift:
            penalty += (required_staff_per_shift - assigned_count) * 10  # Understaffed penalty

    # Check consecutive shifts for each staff
    for staff in range(num_staff):
        for shift in range(num_shifts - 1):
            if schedule[staff][shift] == 1 and schedule[staff][shift + 1] == 1:
                penalty += 5  # Penalty for consecutive shifts

    return penalty

# Function to create a random schedule
def create_random_schedule():
    schedule = []
    for staff in range(num_staff):
        shifts = [0] * num_shifts
        assigned_shifts = []
        
        while len(assigned_shifts) < random.randint(3, max_shifts_per_staff):
            shift = random.randint(0, num_shifts - 1)
            if shift not in assigned_shifts:
                assigned_shifts.append(shift)
                shifts[shift] = 1

        schedule.append(shifts)

    return schedule

# Selection (Top 50%)
def select_parents(population, fitness_scores):
    sorted_population = []
    sorted_fitness = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])

    for i in range(len(fitness_scores) // 2):
        sorted_population.append(population[sorted_fitness[i]])

    return sorted_population

# Crossover (Single point crossover)
def crossover(parent1, parent2):
    child = []
    point = random.randint(1, num_shifts - 1)

    for i in range(num_staff):
        new_staff_shifts = parent1[i][:point] + parent2[i][point:]
        child.append(new_staff_shifts)

    return child

# Mutation (Swap shifts for one staff)
def mutate(schedule):
    staff = random.randint(0, num_staff - 1)
    shift1 = random.randint(0, num_shifts - 1)
    shift2 = random.randint(0, num_shifts - 1)

    # Swap only if both shifts exist
    schedule[staff][shift1], schedule[staff][shift2] = schedule[staff][shift2], schedule[staff][shift1]

    return schedule

# Initialize population
population = []
for i in range(population_size):
    population.append(create_random_schedule())

# Genetic Algorithm loop
for generation in range(max_generations):
    fitness_scores = []
    for schedule in population:
        fitness_scores.append(evaluate_fitness(schedule))

    best_fitness = min(fitness_scores)
    print("Generation", generation + 1, "Best Fitness:", best_fitness)

    parents = select_parents(population, fitness_scores)

    new_population = []
    while len(new_population) < population_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        
        child = crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = mutate(child)

        new_population.append(child)

    population = new_population

# Get the best schedule
best_index = fitness_scores.index(min(fitness_scores))
best_schedule = population[best_index]

# Print the best schedule
print("\nBest Schedule (Staff x Shifts):")
for staff in range(num_staff):
    print("Staff", staff + 1, ":", best_schedule[staff])
