import heapq

start = ("2","8","3","1","-","4","7","6","5")
goal  = ("1","2","3","8","-","4","7","6","5")

def h(s):   # Manhattan distance
    d = 0
    for i,v in enumerate(s):
        if v != "-":
            gi = goal.index(v)
            d += abs(i//3 - gi//3) + abs(i%3 - gi%3)
    return d

moves = {
    0:[1,3], 1:[0,2,4], 2:[1,5],
    3:[0,4,6], 4:[1,3,5,7], 5:[2,4,8],
    6:[3,7], 7:[4,6,8], 8:[5,7]
}

def get_neighbours(s):
    s = list(s)
    i = s.index("-")
    result = []
    for m in moves[i]:
        s[i], s[m] = s[m], s[i]
        result.append(tuple(s))
        s[m], s[i] = s[i], s[m]
    return result

def best_first(start):
    frontier = [(h(start), start)]
    visited = set()

    while frontier:
        _, state = heapq.heappop(frontier)
        
        if state == goal:
            return state
        
        visited.add(state)

        for nb in get_neighbours(state):
            if nb not in visited:
                heapq.heappush(frontier, (h(nb), nb))

