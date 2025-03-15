import random
import math

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(route):
    dist = 0
    for i in range(len(route) - 1):
        dist += distance(route[i], route[i + 1])
    return dist
def hill_climb_delivery_route(locations, max_iterations=1000):

    current_route = locations[:]
    random.shuffle(current_route)
    current_distance = total_distance(current_route)

    for _ in range(max_iterations):
        new_route = current_route[:]
        i, j = random.sample(range(len(locations)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_distance = total_distance(new_route)

        
        if new_distance < current_distance:
            current_route = new_route
            current_distance = new_distance

    return current_route, current_distance

delivery_points = [(2, 3), (5, 8), (1, 1), (7, 2), (4, 6), (9, 9)]

best_route, best_distance = hill_climb_delivery_route(delivery_points)

print("Optimized Delivery Route:", best_route)
print("Total Distance Covered:", best_distance)
