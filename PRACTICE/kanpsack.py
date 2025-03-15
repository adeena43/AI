import random

max_weight = 50

items = [
    (60, 10), (100, 20), (120, 30), (30, 5), (90, 25),
    (50, 15), (70, 10), (80, 20), (20, 10), (40, 5)
]

def fitness(solution):
    total_vallue = 0
    total_weight = 0

    for i in range(solution):
        if solution[i] == 1:
            total_vallue += solution[i][0]
            total_weight += solution[i][1]

    if total_weight > max_weight:
        return 0
        
    return total_vallue
    
def create_random_solution():
    return [random.randint(0, 1) for _ in range(len(items))]

def create_population(size):
    population = []

    for _ in range(size):
        population.append(create_random_solution())

    return population

def select_parents(population):
    sorted_population = sorted(population, key=fitness, reverse=True)
    return sorted_population[:len(population)//2]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(solution, mutation_rate):
    if random.random() < mutation_rate:
        i = random.randint(0, len(solution)-1)
        solution[i] = 1 - solution[i]       # flip the random bit

    return solution

def knapsack(pop_size, generations, mutation_rate):
    population = create_population(pop_size)

    for generation in range(generations):
        parents = select_parents(population)
        new_population = []

        while(len(new_population) < pop_size):
            parent1, parent2 = random.choice(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate)
            new_population.append(child)

        population = new_population
        best_solution = max(new_population)
        best_value = fitness(best_solution)

        print("Generation", generation, "- Best Value:", best_value)
    
    return best_solution, best_value

best_solution, best_value = knapsack(50, 100, 0.1)

print("\nBest Solution:", best_solution)
print("Total Value:", best_value)
print("Items Selected:")
for i in range(len(best_solution)):
    if best_solution[i] == 1:
        print(f"Item {i+1} - Value: {items[i][0]}, Weight: {items[i][1]}")
