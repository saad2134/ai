import random
from collections import deque
import time

class Puzzle7:
    def __init__(self):
        self.size = 3
        self.goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # 0 represents empty space
        self.moves = {
            'up': (-1, 0),
            'down': (1, 0), 
            'left': (0, -1),
            'right': (0, 1)
        }
    
    def create_solvable_puzzle(self, moves=20):
        """Create a solvable puzzle by making random moves from goal state"""
        state = [row[:] for row in self.goal_state]  # Copy goal state
        empty_pos = (2, 2)  # Empty space starts at bottom right
        
        for _ in range(moves):
            possible_moves = []
            for move, (dx, dy) in self.moves.items():
                new_x, new_y = empty_pos[0] + dx, empty_pos[1] + dy
                if 0 <= new_x < self.size and 0 <= new_y < self.size:
                    possible_moves.append((move, (new_x, new_y)))
            
            if possible_moves:
                move, new_pos = random.choice(possible_moves)
                # Swap empty space with adjacent tile
                state[empty_pos[0]][empty_pos[1]] = state[new_pos[0]][new_pos[1]]
                state[new_pos[0]][new_pos[1]] = 0
                empty_pos = new_pos
        
        return state
    
    def find_empty_position(self, state):
        """Find the position of the empty space (0)"""
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return (i, j)
        return None
    
    def get_possible_moves(self, state):
        """Get all possible moves from current state"""
        empty_pos = self.find_empty_position(state)
        possible_states = []
        
        for move, (dx, dy) in self.moves.items():
            new_x, new_y = empty_pos[0] + dx, empty_pos[1] + dy
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                # Create new state by swapping empty space with adjacent tile
                new_state = [row[:] for row in state]  # Copy state
                new_state[empty_pos[0]][empty_pos[1]] = new_state[new_x][new_y]
                new_state[new_x][new_y] = 0
                possible_states.append((move, new_state))
        
        return possible_states
    
    def is_goal_state(self, state):
        """Check if current state is the goal state"""
        return state == self.goal_state
    
    def manhattan_distance(self, state):
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] != 0:
                    value = state[i][j]
                    goal_i, goal_j = (value - 1) // self.size, (value - 1) % self.size
                    distance += abs(i - goal_i) + abs(j - goal_j)
        return distance
    
    def print_state(self, state):
        """Print the puzzle state in a nice format"""
        print("+---+---+---+")
        for row in state:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("   |", end="")
                else:
                    print(f" {cell} |", end="")
            print("\n+---+---+---+")
    
    def bfs_solve(self, initial_state):
        """Solve using Breadth-First Search"""
        queue = deque([(initial_state, [])])
        visited = set()
        visited.add(self.state_to_tuple(initial_state))
        
        nodes_explored = 0
        
        while queue:
            current_state, path = queue.popleft()
            nodes_explored += 1
            
            if self.is_goal_state(current_state):
                return path, nodes_explored
            
            for move, next_state in self.get_possible_moves(current_state):
                state_tuple = self.state_to_tuple(next_state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    queue.append((next_state, path + [move]))
        
        return None, nodes_explored
    
    def a_star_solve(self, initial_state):
        """Solve using A* algorithm with Manhattan distance heuristic"""
        open_set = [(self.manhattan_distance(initial_state), 0, initial_state, [])]
        visited = set()
        visited.add(self.state_to_tuple(initial_state))
        
        nodes_explored = 0
        
        while open_set:
            open_set.sort(key=lambda x: x[0] + x[1])  # Sort by f = g + h
            f_cost, g_cost, current_state, path = open_set.pop(0)
            nodes_explored += 1
            
            if self.is_goal_state(current_state):
                return path, nodes_explored
            
            for move, next_state in self.get_possible_moves(current_state):
                state_tuple = self.state_to_tuple(next_state)
                if state_tuple not in visited:
                    visited.add(state_tuple)
                    new_g_cost = g_cost + 1
                    new_f_cost = new_g_cost + self.manhattan_distance(next_state)
                    open_set.append((new_f_cost, new_g_cost, next_state, path + [move]))
        
        return None, nodes_explored
    
    def state_to_tuple(self, state):
        """Convert state to tuple for hashing"""
        return tuple(tuple(row) for row in state)
    
    def interactive_game(self):
        """Play the 7 puzzle game interactively"""
        print("=== 7 Puzzle Game ===")
        print("Goal State:")
        self.print_state(self.goal_state)
        print("\nInstructions:")
        print("• Use WASD keys to move the empty space")
        print("• W = Up, A = Left, S = Down, D = Right")
        print("• Type 'solve' to let AI solve the puzzle")
        print("• Type 'quit' to exit\n")
        
        current_state = self.create_solvable_puzzle(15)  # Create easier puzzle
        moves_made = 0
        
        while True:
            print(f"\nMove {moves_made}")
            print("Current State:")
            self.print_state(current_state)
            
            if self.is_goal_state(current_state):
                print(f"\n🎉 Congratulations! You solved the puzzle in {moves_made} moves!")
                break
            
            move = input("\nYour move (WASD/solve/quit): ").lower().strip()
            
            if move == 'quit':
                print("Thanks for playing!")
                break
            elif move == 'solve':
                self.ai_solve_demo(current_state)
                break
            else:
                # Convert WASD to move directions
                move_map = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
                if move in move_map:
                    move_direction = move_map[move]
                    moved = False
                    
                    for possible_move, next_state in self.get_possible_moves(current_state):
                        if possible_move == move_direction:
                            current_state = next_state
                            moves_made += 1
                            moved = True
                            break
                    
                    if not moved:
                        print("Invalid move! Try again.")
                else:
                    print("Invalid input! Use W, A, S, D, 'solve', or 'quit'")
    
    def ai_solve_demo(self, puzzle_state):
        """Demonstrate AI solving the puzzle"""
        print("\n🤖 AI Solver Demonstration")
        print("Initial puzzle:")
        self.print_state(puzzle_state)
        
        # Solve with BFS
        print("\n1. Solving with BFS...")
        start_time = time.time()
        bfs_solution, bfs_nodes = self.bfs_solve(puzzle_state)
        bfs_time = time.time() - start_time
        
        if bfs_solution:
            print(f"✅ BFS Solution found in {len(bfs_solution)} moves")
            print(f"Nodes explored: {bfs_nodes}")
            print(f"Time: {bfs_time:.3f} seconds")
            print(f"Solution: {' → '.join(bfs_solution)}")
        else:
            print("❌ BFS could not find solution")
        
        # Solve with A*
        print("\n2. Solving with A*...")
        start_time = time.time()
        astar_solution, astar_nodes = self.a_star_solve(puzzle_state)
        astar_time = time.time() - start_time
        
        if astar_solution:
            print(f"✅ A* Solution found in {len(astar_solution)} moves")
            print(f"Nodes explored: {astar_nodes}")
            print(f"Time: {astar_time:.3f} seconds")
            print(f"Solution: {' → '.join(astar_solution)}")
            
            # Show solution steps
            if input("\nShow solution steps? (y/n): ").lower() == 'y':
                current = [row[:] for row in puzzle_state]
                print("\nSolution Steps:")
                self.print_state(current)
                
                for step, move in enumerate(astar_solution, 1):
                    input(f"\nStep {step}: Press Enter to see {move} move...")
                    for possible_move, next_state in self.get_possible_moves(current):
                        if possible_move == move:
                            current = next_state
                            break
                    self.print_state(current)
                
                print("🎉 Puzzle solved!")
        else:
            print("❌ A* could not find solution")
        
        # Compare algorithms
        if bfs_solution and astar_solution:
            print(f"\n📊 Algorithm Comparison:")
            print(f"BFS:  {len(bfs_solution)} moves, {bfs_nodes} nodes, {bfs_time:.3f}s")
            print(f"A*:   {len(astar_solution)} moves, {astar_nodes} nodes, {astar_time:.3f}s")
            print(f"A* explored {bfs_nodes/astar_nodes:.1f}x fewer nodes!")

def benchmark_solver():
    """Benchmark the solver on multiple puzzles"""
    puzzle = Puzzle7()
    print("=== Solver Benchmark ===")
    
    puzzles = []
    for difficulty in [10, 15, 20]:
        puzzle_state = puzzle.create_solvable_puzzle(difficulty)
        puzzles.append((f"{difficulty} moves", puzzle_state))
    
    for name, puzzle_state in puzzles:
        print(f"\n{name}:")
        
        # BFS
        start_time = time.time()
        bfs_solution, bfs_nodes = puzzle.bfs_solve(puzzle_state)
        bfs_time = time.time() - start_time
        
        # A*
        start_time = time.time()
        astar_solution, astar_nodes = puzzle.a_star_solve(puzzle_state)
        astar_time = time.time() - start_time
        
        print(f"BFS:  {len(bfs_solution) if bfs_solution else 'N/A'} moves, {bfs_nodes:5d} nodes, {bfs_time:.3f}s")
        print(f"A*:   {len(astar_solution) if astar_solution else 'N/A'} moves, {astar_nodes:5d} nodes, {astar_time:.3f}s")

if __name__ == "__main__":
    puzzle = Puzzle7()
    
    print("Choose mode:")
    print("1. Play interactive game")
    print("2. AI solver demonstration") 
    print("3. Benchmark solver")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        puzzle.interactive_game()
    elif choice == "2":
        test_puzzle = puzzle.create_solvable_puzzle(15)
        puzzle.ai_solve_demo(test_puzzle)
    elif choice == "3":
        benchmark_solver()
    else:
        print("Starting interactive game...")
        puzzle.interactive_game()