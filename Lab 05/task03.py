import math
import random

def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def total_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(route[i], route[i + 1])
    dist += distance(route[-1], route[0])
    return dist

def fitness(route):
    return 1 / total_distance(route)

def create_random_route(cities):
    route = cities[:]
    random.shuffle(route)
    return route

def create_population(cities, pop_size):
    population = []
    for i in range(pop_size):
        population.append(create_random_route(cities))
    return population

def select_parents(population, num_parents):
    parents = []
    sorted_population = sorted(population, key=total_distance)
    for i in range(num_parents):
        parents.append(sorted_population[i])
    return parents

def crossover(parent1, parent2):
    point = random.randint(0, len(parent1) - 1)
    child = parent1[:point]
    for city in parent2:
        if city not in child:
            child.append(city)
    return child

def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        i = random.randint(0, len(route) - 1)
        j = random.randint(0, len(route) - 1)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(cities, population_size, generations, mutation_rate):
    population = create_population(cities, population_size)

    for generation in range(generations):
        parents = select_parents(population, population_size // 2)
        new_population = []

        for i in range(population_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        best_route = min(population, key=total_distance)
        best_dist = total_distance(best_route)

        print("Generation", generation, "- Best Distance:", best_dist)

    return best_route, best_dist

cities = []
for i in range(10):
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    cities.append((x, y))

best_route, best_distance = genetic_algorithm(cities, population_size=50, generations=100, mutation_rate=0.1)

print("\nBest Route:", best_route)
print("Total Distance:", best_distance)
