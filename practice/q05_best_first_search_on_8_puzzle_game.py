import heapq

start = ("2", "8", "3", "1", "-", "4", "7", "6", "5")
goal = ("1", "2", "3", "8", "-", "4", "7", "6", "5")

# Manhattan Distance
def h(state):
    d = 0
    for i, v in enumerate(state):
        if v != "-":  # Ignore the blank tile
            gi = goal.index(v)
            d += abs(i // 3 - gi // 3) + abs(i % 3 - gi % 3)  # Accumulate the distance
    return d

# Possible moves of blank (the positions the blank tile can move to)
moves = {
    0: [1, 3],   1: [0, 2, 4],   2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7],   7: [4, 6, 8],   8: [5, 7]
}

def neighbours(state):
    s = list(state)
    i = s.index("-")  # Index of the blank tile
    for m in moves[i]:
        s[i], s[m] = s[m], s[i]  # Swap the blank tile with the neighbor
        yield tuple(s)  # Return the new state
        s[m], s[i] = s[i], s[m]  # Swap back to generate other neighbors

def best_first(start, goal):
    frontier = [(h(start), start, [])]  # Priority queue of (heuristic, state, path)
    seen = set()  # Set of visited states
    while frontier:
        _, state, path = heapq.heappop(frontier)  # Get state with lowest heuristic
        if state in seen:
            continue
        seen.add(state)
        path = path + [state]  # Add current state to path
        if state == goal:
            return path  # Return the path when goal is reached
        for nb in neighbours(state):
            if nb not in seen:
                heapq.heappush(frontier, (h(nb), nb, path))  # Add neighbors to frontier

# Run the best-first search
solution = best_first(start, goal)

# Show Steps
for step, s in enumerate(solution):
    print("Step ", step, " ( h =", h(s), ")")
    for i in range(0, 9, 3):
        print(" ".join(s[i:i + 3]))  # Print the state in a 3x3 grid format

