import random

# Define the number of queens
n = 8

# Function to calculate fitness (number of non-attacking queen pairs)
def calculate_fitness(individual):
    non_attacking_pairs = 0
    total_pairs = (n * (n - 1)) // 2  # Maximum possible non-attacking pairs

    # Compare each pair of queens
    for i in range(n):
        for j in range(i + 1, n):  
            # Check if they are NOT attacking each other
            if individual[i] != individual[j] and abs(individual[i] - individual[j]) != abs(i - j):
                non_attacking_pairs += 1

    # Return fitness score (closer to 1 means better solution)
    return non_attacking_pairs / total_pairs

# Function to create a random board arrangement (solution candidate)
def create_random_individual():
    numbers = list(range(n))  # Create a list of numbers [0,1,2,...,n-1]
    random.shuffle(numbers)  # Shuffle to ensure uniqueness
    return numbers

# Function to create an initial population
def create_initial_population(size):
    population = []
    for _ in range(size):
        population.append(create_random_individual())
    return population

# Function to evaluate the fitness of each individual in the population
def evaluate_population(population):
    fitness_scores = []
    for individual in population:
        fitness_scores.append(calculate_fitness(individual))
    return fitness_scores

# Function to select the top individuals as parents
def select_parents(population, fitness_scores):
    sorted_population = []  

    # Pair each individual with its fitness score
    for i in range(len(population)):
        sorted_population.append((fitness_scores[i], population[i]))

    # Sort by fitness (higher is better)
    sorted_population.sort(reverse=True, key=lambda x: x[0])

    # Select the top 50% of the population as parents
    selected_parents = []
    for i in range(len(population) // 2):
        selected_parents.append(sorted_population[i][1])  # Only keep individuals (not scores)

    return selected_parents

# Function to perform crossover (mix two parents to create a child)
def crossover(parent1, parent2):
    point = random.randint(1, n - 2)  # Random crossover point

    # Take first part from parent1, second part from parent2
    child = parent1[:point] + parent2[point:]

    # Ensure uniqueness (replace duplicates with missing numbers)
    missing_numbers = list(set(range(n)) - set(child))
    for i in range(len(child)):
        if child.count(child[i]) > 1:  # If duplicate
            child[i] = missing_numbers.pop(0)

    return child

# Function to apply mutation (swap two positions randomly)
def mutate(individual):
    idx1 = random.randint(0, n - 1)
    idx2 = random.randint(0, n - 1)

    # Swap two values
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    return individual

# Function to apply genetic algorithm
def genetic_algorithm():
    population_size = 10
    mutation_rate = 0.1
    generations = 100
    population = create_initial_population(population_size)
    
    for generation in range(generations):
        fitness_scores = evaluate_population(population)
        best_fitness = max(fitness_scores)

        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # If perfect solution found, stop early
        if best_fitness == 1.0:
            break

        # Select parents
        parents = select_parents(population, fitness_scores)

        # Create new population using crossover
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            new_population.append(child)

        # Apply mutation
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                new_population[i] = mutate(new_population[i])

        # Update population
        population = new_population

    # Return the best solution
    best_solution = max(population, key=calculate_fitness)
    return best_solution, calculate_fitness(best_solution)

# Run the genetic algorithm
solution, fitness = genetic_algorithm()

print("\nBest Solution:", solution)
print("Best Fitness Score:", fitness)
