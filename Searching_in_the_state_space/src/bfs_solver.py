from collections import deque
from src.core_logic import get_start_goal, get_neighbors, print_maze_with_path

def bfs(maze):
    start, goal = get_start_goal(maze)
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current_node, path = queue.popleft() 
        
        if current_node == goal:
            print_maze_with_path(maze, path, len(visited))
            return path
            
        for neighbor in get_neighbors(maze, current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    print("Không tìm thấy đường đi!")
    return None