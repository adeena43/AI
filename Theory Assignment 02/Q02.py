import random
import numpy as np

tasks_time = [5, 8, 4, 7, 6, 3, 9]
facilities_cap = [24, 30, 28]
cost_matrix = [
    [10, 12, 9],
    [15, 14, 16],
    [8, 9, 7],
    [12, 10, 13],
    [14, 13, 12],
    [9, 8, 10],
    [11, 12, 13]
]

POP_SIZE = 6
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
MAX_GENERATIONS = 100

def create_chromosome():
    return [random.randint(0, 2) for _ in range(7)]

def calculate_fitness(chromosome):
    total_cost = 0
    facility_usage = [0, 0, 0]
    penalty = 0
    
    for task, facility in enumerate(chromosome):
        time = tasks_time[task]
        cost = cost_matrix[task][facility]
        total_cost += time * cost
        facility_usage[facility] += time
    

    for fac in range(3):
        if facility_usage[fac] > facilities_cap[fac]:
            penalty += (facility_usage[fac] - facilities_cap[fac]) * 1000
    
    return total_cost + penalty

def roulette_selection(population, fitnesses):
    inverse_fitness = [1/f for f in fitnesses]
    total = sum(inverse_fitness)
    probs = [f/total for f in inverse_fitness]
    return random.choices(population, weights=probs, k=2)

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, 5)
        child = parent1[:point] + parent2[point:]
        return child
    return parent1 if random.random() < 0.5 else parent2

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(7), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def genetic_algorithm():
    population = [create_chromosome() for _ in range(POP_SIZE)]
    
    for generation in range(MAX_GENERATIONS):
        fitnesses = [calculate_fitness(chrom) for chrom in population]
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_chrom = population[best_idx]
        
        print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        new_population = [best_chrom]  
        
        while len(new_population) < POP_SIZE:
            parents = roulette_selection(population, fitnesses)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)
        
        population = new_population

    fitnesses = [calculate_fitness(chrom) for chrom in population]
    best_idx = np.argmin(fitnesses)
    best_chrom = population[best_idx]

    allocation = {0: [], 1: [], 2: []}
    total_cost = 0
    for task, fac in enumerate(best_chrom):
        allocation[fac].append(task+1)
        total_cost += tasks_time[task] * cost_matrix[task][fac]
    
    print("\nFinal Solution:")
    for fac in allocation:
        print(f"Facility {fac+1}: Tasks {allocation[fac]}")
    print(f"Total Cost: {total_cost}")

genetic_algorithm()