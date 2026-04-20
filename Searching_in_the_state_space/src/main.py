from maps.maze_basic import maze as basic_maze
from maps.maze_hard import maze as hard_maze
from src.bfs_solver import bfs
from src.dfs_solver import dfs

if __name__ == "__main__":
    print("\n========== BẢN ĐỒ CƠ BẢN ==========")
    print(">>> CHẠY THUẬT TOÁN BFS:")
    bfs(basic_maze)
    
    print("\n>>> CHẠY THUẬT TOÁN DFS:")
    dfs(basic_maze)
    
    print("\n========== BẢN ĐỒ KHÓ (10x10) ==========")
    print(">>> CHẠY THUẬT TOÁN BFS:")
    bfs(hard_maze)
    
    print("\n>>> CHẠY THUẬT TOÁN DFS:")
    dfs(hard_maze)