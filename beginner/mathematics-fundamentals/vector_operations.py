import math

def vector_operations(v1, v2):
    add = [v1[i] + v2[i] for i in range(len(v1))]
    dot = sum(v1[i] * v2[i] for i in range(len(v1)))
    mag = math.sqrt(sum(x**2 for x in v1))
    return add, dot, mag

v1 = [1, 2, 3]
v2 = [4, 5, 6]
add, dot, mag = vector_operations(v1, v2)
print(f"Addition: {add}, Dot: {dot}, Magnitude: {mag:.2f}")