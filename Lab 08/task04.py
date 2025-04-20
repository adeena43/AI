import numpy as np

states = ['Sunny', 'Cloudy', 'Rainy']
state_to_index = {state: i for i, state in enumerate(states)}

transition_matrix = [
    [0.6, 0.3, 0.1],  # From Sunny
    [0.3, 0.4, 0.3],  # From Cloudy
    [0.2, 0.3, 0.5]   # From Rainy
]
def simulate_weather(start_state, days=10):
    current_state = state_to_index[start_state]
    weather_sequence = [start_state]

    for _ in range(days - 1):
        next_state = np.random.choice(
            states,
            p=transition_matrix[current_state]
        )
        weather_sequence.append(next_state)
        current_state = state_to_index[next_state]

    return weather_sequence

def estimate_rainy_probability(simulations=10000):
    rainy_counts = 0
    for _ in range(simulations):
        forecast = simulate_weather('Sunny', days=10)
        rainy_days = forecast.count('Rainy')
        if rainy_days >= 3:
            rainy_counts += 1
    probability = rainy_counts / simulations
    return probability

sample_sequence = simulate_weather('Sunny', 10)
print("Sample 10-day weather forecast:", sample_sequence)
rainy_probability = estimate_rainy_probability()
print(f"\nEstimated Probability of >= 3 rainy days in 10 days: {rainy_probability:.4f}")
