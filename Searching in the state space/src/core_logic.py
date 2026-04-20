def get_start_goal(maze):
    start = goal = None
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == 'S': start = (r, c)
            if maze[r][c] == 'G': goal = (r, c)
    return start, goal

def get_neighbors(maze, node):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Lên, Xuống, Trái, Phải
    neighbors = []
    r, c = node
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]) and maze[nr][nc] != 1:
            neighbors.append((nr, nc))
    return neighbors

def print_maze_with_path(maze, path, visited_count):
    print(f"\n--- TÌM THẤY ĐÍCH! Đã duyệt qua {visited_count} ô ---")
    maze_copy = [row[:] for row in maze]
    
    # Đánh dấu đường đi bằng dấu '+'
    if path:
        for r, c in path:
            if maze_copy[r][c] not in ['S', 'G']:
                maze_copy[r][c] = '+'
                
    for row in maze_copy:
        row_str = ""
        for cell in row:
            if cell == 1: row_str += "██"      # Tường
            elif cell == 0: row_str += " . "     # Đường đi
            elif cell == '+': row_str += " + "   # Đường robot đi qua
            elif cell == 'S': row_str += " S "   # Start
            elif cell == 'G': row_str += " G "   # Goal
        print(row_str)
    print("-" * 40)