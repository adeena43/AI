class Product:
    def __init__(self, name, frequency, volume):
        self.name = name
        self.frequency = frequency
        self.volume = volume

class Slot:
    def __init__(self, name, distance, capacity):
        self.name = name
        self.distance = distance
        self.capacity = capacity
        self.remaining_capacity = capacity

def is_valid_assignment(product, slot):
    # Check if the slot has enough capacity for the product
    return slot.remaining_capacity >= product.volume

def assign_product(product, slot):
    # Assign the product to the slot and reduce the slot's remaining capacity
    slot.remaining_capacity -= product.volume

def unassign_product(product, slot):
    # Unassign the product from the slot and restore the slot's remaining capacity
    slot.remaining_capacity += product.volume

def backtrack(products, slots, assignment, index):
    if index == len(products):
        # All products have been assigned
        return True

    product = products[index]

    # Try assigning the product to each slot in order of increasing distance
    for slot in sorted(slots, key=lambda x: x.distance):
        if is_valid_assignment(product, slot):
            # Assign the product to the slot
            assign_product(product, slot)
            assignment[product.name] = slot.name

            # Recur to assign the next product
            if backtrack(products, slots, assignment, index + 1):
                return True

            # Backtrack if the assignment leads to a dead end
            unassign_product(product, slot)
            del assignment[product.name]

    return False

def solve_warehouse_csp(products, slots):
    # Sort products by frequency in descending order (most frequent first)
    products_sorted = sorted(products, key=lambda x: x.frequency, reverse=True)

    # Initialize assignment dictionary
    assignment = {}

    # Start backtracking
    if backtrack(products_sorted, slots, assignment, 0):
        return assignment
    else:
        return None

# Sample Input
products = [
    Product(name="Product1", frequency=15, volume=2),
    Product(name="Product2", frequency=8, volume=1),
    Product(name="Product3", frequency=20, volume=3)
]

slots = [
    Slot(name="Slot1", distance=1, capacity=2),
    Slot(name="Slot2", distance=2, capacity=3),
    Slot(name="Slot3", distance=3, capacity=4)
]

# Solve the CSP
assignment = solve_warehouse_csp(products, slots)

# Output the assignment
if assignment:
    print("Product Assignments:")
    for product, slot in assignment.items():
        print(f"{product} is assigned to {slot}")
else:
    print("No valid assignment found.")
