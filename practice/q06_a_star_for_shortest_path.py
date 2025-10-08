import heapq

def a_star(graph, start, goal, h):
    open_set = []
    heapq.heappush(open_set,(0 + h[start], 0, start, [start]))
                   
    visited = set()
                   
    while open_set:
        estimated_cost, cost_so_far, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, cost_so_far

        if current in visited:
            continue

        visited.add(current)

        for neighbour, weight in graph[current].items():
            if neighbour not in visited:
                total_cost = cost_so_far + weight
                estimated_total = total_cost + h[neighbour]
                heapq.heappush(open_set, (estimated_total, total_cost, neighbour, path + [neighbour]))
    
    return None, float('inf')

#example graph
graph = {
    'A': {'B':1, 'C':4 },
    'B': {'A':1, 'C':2, 'D':5 },
    'C': {'A':4, 'B':2, 'D':1 },
    'D': {'B':5, 'C':1 }
}

heuristic = {
    'A': 7,
    'B': 6,
    'C': 2,
    'D': 0
}

start_node = 'A'
goal_node = 'D'

path, cost = a_star(graph, start_node, goal_node ,heuristic)
print(f"Shortest Path: {path}")
print(f"Total Cost: {cost}")


    
    
