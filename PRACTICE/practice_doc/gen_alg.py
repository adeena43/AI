import random

# Function to calculate f(x) = 2x² - 1
def fitness(x):
    return 2 * (x ** 2) - 1

# Convert binary string to integer
def binary_to_int(binary_str):
    return int(binary_str, 2)

# Convert integer to 6-bit binary string
def int_to_binary(x):
    return format(x, '06b')

# Generate initial population
def generate_population(size):
    population = []
    for _ in range(size):
        x = random.randint(0, 31)  # Random number between 0 and 31
        population.append(int_to_binary(x))  # Convert to 6-bit binary
    return population

# Tournament selection
def tournament_selection(population, k=3):
    tournament = random.sample(population, k)  # Pick k random individuals
    best_individual = tournament[0]
    best_fitness = fitness(binary_to_int(best_individual))

    for ind in tournament:
        ind_fitness = fitness(binary_to_int(ind))
        if ind_fitness > best_fitness:
            best_fitness = ind_fitness
            best_individual = ind

    return best_individual

# Uniform crossover
def uniform_crossover(parent1, parent2):
    child = ""
    for i in range(len(parent1)):  # Loop through each bit
        if random.random() < 0.5:  # 50% chance to take from parent1
            child += parent1[i]
        else:
            child += parent2[i]
    return child

# Adaptive mutation
def adaptive_mutation(individual, generation, max_generations):
    mutation_rate = max(0.05, 0.2 * (1 - generation / max_generations))  # Reduce over time
    mutated = ""

    for bit in individual:  # Loop through each bit
        if random.random() < mutation_rate:  # Randomly mutate bit
            mutated += '1' if bit == '0' else '0'
        else:
            mutated += bit
    return mutated

# Genetic Algorithm function
def genetic_algorithm(pop_size=20, generations=50):
    population = generate_population(pop_size)
    best_solution = None

    for gen in range(generations):
        new_population = []

        for _ in range(pop_size):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = uniform_crossover(parent1, parent2)
            child = adaptive_mutation(child, gen, generations)
            new_population.append(child)

        population = new_population  # Replace old population

        # Find the best solution in current generation
        best_individual = population[0]
        best_fitness = fitness(binary_to_int(best_individual))

        for ind in population:
            ind_fitness = fitness(binary_to_int(ind))
            if ind_fitness > best_fitness:
                best_individual = ind
                best_fitness = ind_fitness

        if best_solution is None or best_fitness > fitness(binary_to_int(best_solution)):
            best_solution = best_individual

        # Print best solution per generation
        print(f"Generation {gen + 1}: Best x = {binary_to_int(best_solution)}, f(x) = {fitness(binary_to_int(best_solution))}")

    return best_solution

# Run the algorithm
best_binary = genetic_algorithm()
best_x = binary_to_int(best_binary)
best_fitness = fitness(best_x)

print(f"\nBest Solution Found: x = {best_x}, Binary = {best_binary}, f(x) = {best_fitness}")
