

graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}





#BREADTH FIRST SEARCH

visitedbfs = [] #list for visited nodes
queue = [] #initialize a queue

def bfs(visited, graph, node): #function for BFS
    visited.append(node)
    queue.append(node)
    while queue: #Creating loop to visit each node
        m = queue.pop()
        print(m, end="\n")
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)

#Driver Code
print("Following is the Breadth First Search: ")
bfs(visitedbfs, graph, '5')





#DEPTH FIRST SEARCH

visiteddfs = set() #set to keep track of visited node of each graph

def dfs(visited, graph, node): #function for DFS
    if node not in visited:
        print(node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

#Driver Code
print("Following is the Depth First Search: ")
dfs(visiteddfs, graph, '5')
