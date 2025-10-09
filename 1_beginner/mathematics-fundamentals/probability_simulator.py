import random
import collections

def binomial_simulation(n_flips=1000):
    results = [sum(random.choice([0, 1]) for _ in range(10)) for _ in range(n_flips)]
    freq = collections.Counter(results)
    for heads, count in sorted(freq.items()):
        print(f"Heads: {heads}, Probability: {count/n_flips:.3f}")

binomial_simulation()