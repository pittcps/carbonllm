import random

def generate_random_pairs(n, min_val, max_val, filename, seed=None):
    random.seed(seed)
    with open(filename, 'w') as file:
        for _ in range(n):
            pair = (random.randint(min_val, max_val), random.randint(min_val, max_val))
            file.write(f"{pair[0]} {pair[1]}\n")

n = 100  # Number of pairs
min_val = 1  # Minimum value for random integers
max_val = 10000  # Maximum value for random integers
filename = "../output/100_random_seeds.txt"  # Name of the file to save pairs
seed = 42

generate_random_pairs(n, min_val, max_val, filename, seed)
print(f"{n} pairs of random integers saved to '{filename}'.")
