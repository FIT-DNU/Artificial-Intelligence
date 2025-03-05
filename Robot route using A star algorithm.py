from typing import List, Tuple, Dict, Set
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt

def create_node(position: Tuple[int, int], g: float = float('inf'), h: float = 0.0, parent: Dict = None) -> Dict:
    """
    Tạo một nút cho thuật toán A*.
    Tham số:
    - position: Tọa độ (x, y) của nút
    - g: Chi phí từ điểm bắt đầu đến nút này (mặc định: vô cùng)
    - h: Chi phí ước lượng từ nút này đến mục tiêu (mặc định: 0)
    - parent: Nút cha (mặc định: None)
    Trả về:
    - Từ điển chứa thông tin nút
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Tính toán khoảng cách ước lượng giữa hai điểm bằng khoảng cách Euclid.
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Lấy tất cả các vị trí láng giềng hợp lệ trong lưới.
    Tham số:
    - grid: Mảng 2D numpy, 0 là ô có thể đi qua, 1 là chướng ngại vật,
      2 là bùn lầy, 3 là tường đá
    - position: Vị trí hiện tại (x, y)
    Trả về:
    - Danh sách các vị trí láng giềng hợp lệ
    """
    x, y = position
    rows, cols = grid.shape
    # Các hướng di chuyển khả thi (bao gồm đường chéo)
    possible_moves = [
        (x+1, y), (x-1, y), # Phải, Trái
        (x, y+1), (x, y-1), # Lên, Xuống
    ]
    neighbors = []
    for nx, ny in possible_moves:
        if 0 <= nx < rows and 0 <= ny < cols: # Trong giới hạn lưới
            if grid[nx, ny] != 1: # Không phải chướng ngại vật
                neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    """
    Tái tạo đường đi từ mục tiêu về điểm bắt đầu bằng cách theo dấu các nút cha.
    """
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']

    return path[::-1] # Đảo ngược để có đường đi từ đầu đến cuối


def find_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Tìm đường đi tối ưu sử dụng thuật toán A*.
    Tham số:
    - grid: Mảng 2D numpy (0 = ô trống, 1 = chướng ngại vật)
    - start: Vị trí bắt đầu (x, y)
    - goal: Vị trí mục tiêu (x, y)
    Trả về:
    - Danh sách các vị trí đại diện cho đường đi tối ưu
    """
    # Khởi tạo nút bắt đầu
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )
    # Khởi tạo danh sách mở và đóng
    open_list = [(start_node['f'], start)] # Hàng đợi ưu tiên
    open_dict = {start: start_node} # Tra cứu nhanh các nút
    closed_set = set() # Các nút đã khám phá

    while open_list:
        # Lấy nút có giá trị f thấp nhất
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # Kiểm tra nếu đã đến mục tiêu
        if current_pos == goal:
            return reconstruct_path(current_node)

        closed_set.add(current_pos)

        # Khám phá các láng giềng
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Bỏ qua nếu đã khám phá
            if neighbor_pos in closed_set:
                continue

            # Tính toán chi phí đường đi mới, phụ thuộc vào loại ô
            x, y = neighbor_pos
            if grid[x, y] == 2:  # Bùn lầy
                tentative_g = current_node['g'] + 3  # Chi phí bùn lầy
            elif grid[x, y] == 3:  # Tường đá
                tentative_g = current_node['g'] + 5  # Chi phí tường đá
            else:  # Ô trống
                tentative_g = current_node['g'] + 1  # Chi phí ô trống

            # Tạo hoặc cập nhật láng giềng
            if neighbor_pos not in open_dict:
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Tìm được đường đi tốt hơn tới láng giềng
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node

    return []  # Không tìm thấy đường đi

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]) -> None:
    """
    Hiển thị lưới với đường đi được đánh dấu bằng '*'.
    Tham số:
    - grid: Mảng 2D numpy đại diện cho kho
    - path: Danh sách các vị trí đại diện cho đường đi
    """
    # Tạo bản sao của lưới để tránh thay đổi dữ liệu gốc
    grid_copy = np.copy(grid)

    # Đánh dấu đường đi trên lưới
    for (x, y) in path:
        grid_copy[x][y] = 8 # Sử dụng 8 để biểu diễn đường đi

    # In ra lưới
    for row in grid_copy:
        print(' '.join(['*' if cell == 8 else str(cell) for cell in row]))

def plot_grid(grid: np.ndarray, path: List[Tuple[int, int]]) -> None:
    """
    Vẽ lưới và đường đi sử dụng matplotlib.
    Tham số:
    - grid: Mảng 2D numpy đại diện cho kho
    - path: Danh sách các vị trí đại diện cho đường đi
    """
    # Tạo hình và trục
    fig, ax = plt.subplots()

    # Vẽ lưới
    ax.imshow(grid, cmap='Greys', interpolation='none')

    # Vẽ đường đi
    if path:
        path_x = [p[1] for p in path] # Cột (y)
        path_y = [p[0] for p in path] # Hàng (x)
        ax.plot(path_x, path_y, color='red', marker='o', markersize=8, linewidth=2, label='Path')

    # Đánh dấu vị trí bắt đầu và mục tiêu
    start = path[0] if path else None
    goal = path[-1] if path else None

    if start:
        ax.plot(start[1], start[0], color='green', marker='s', markersize=10, label='Start')
    if goal:
        ax.plot(goal[1], goal[0], color='blue', marker='s', markersize=10, label='Goal')

    # Thêm chú giải
    ax.legend()
    # Hiển thị đồ thị
    plt.show()

def main():
    grid = np.zeros((20, 20)) # Lưới 20x20, ban đầu toàn bộ là ô trống
    #grid[0:19,0:19]=1

    # Thêm một số chướng ngại vật
    # grid[5:15, 10] = 1 # Tường dọc
    # grid[5, 5:15] = 1 # Tường ngang

    # grid[5:15, 10] = 1 # Vertical wall
    grid[10, 5:15] = 3 # Horizontal wall
    grid[9, 3:5] = 3
    grid[8, 0:3] = 3
    grid[11, 15:19] = 2
    grid[13:17, 15] = 2
    # Định nghĩa vị trí bắt đầu và mục tiêu
    start_pos = (2, 2)
    goal_pos = (18, 18)

    # Tìm đường đi
    path = find_path(grid, start_pos, goal_pos)
    if path:
        print(f"Đường đi được tìm thấy với {len(path)} bước!")
        visualize_path(grid, path)
        plot_grid(grid, path)
    else:
        print("Không tìm thấy đường đi!")

if __name__ == "__main__":
    main()
