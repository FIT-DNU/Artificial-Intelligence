from src.core_logic import get_start_goal, get_neighbors, print_maze_with_path

def dfs(maze):
    start, goal = get_start_goal(maze)
    stack = [(start, [start])]
    visited = set([start])
    
    while stack:
        current_node, path = stack.pop() 
        
        if current_node == goal:
            print_maze_with_path(maze, path, len(visited))
            return path
            
        for neighbor in get_neighbors(maze, current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
                
    print("Không tìm thấy đường đi!")
    return None