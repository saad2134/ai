import math

def logistic(z):
    return 1 / (1 + math.exp(-z))

for z in range(-5, 6):
    sigma = logistic(z)
    print(f"σ({z}) = {sigma:.4f}")