import heapq

start = ("2", "8", "3", "1", "-", "4", "7", "6", "5")
goal = ("1", "2", "3", "8", "-", "4", "7", "6", "5")

#Manhattan Distance
def h(state):
    d = 0
    for i, v in enumerate(state):
        if v != "_" and v in goal:
            gi = goal.index(v)
            d = (abs(i//3 - gi//3) + abs(i%3 - gi%3))
    return d

#Possible moves of blank
moves = {
    0:[1,3],   1:[0,2,4],   2:[1,5],
    3:[0,4,6], 4:[1,3,5,7], 5:[2,4,8],
    6:[3,7],   7:[4,6,8],   8:[5,7]
}

def neighbours(state):
    s = list(state)
    i = s.index("-")
    for m in moves[i]:
        s[i],s[m]=s[m],s[i]
        yield tuple(s)
        s[m],s[i]=s[i],s[m]

def best_first(start, goal):
    frontier = [(h(start), start,[])]
    seen = set()
    while frontier:
        _, state, path = heapq.heappop(frontier)
        if state in seen:
            continue
        seen.add(state)
        path=path+[state]
        if state == goal:
            return path
        for nb in neighbours(state):
            if nb not in seen:
                heapq.heappush(frontier, (h(nb),nb,path))

#Run
solution = best_first(start, goal)

#Show Steps
for step, s in enumerate(solution):
    print("Step ",step," ( h =", h(s),")")
    for i in range(0,9,3):
        print(" ".join(s[i:i+3]))

print()
