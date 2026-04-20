# Bài Lab Trí tuệ Nhân tạo: Robot Cứu hộ PCCC 🚒

Chào mừng các em đến với bài Lab đầu tiên. Trong bài này, chúng ta sẽ mô hình hóa bài toán dẫn đường cho một Robot cứu hộ hỏa hoạn tìm nạn nhân trong tòa nhà bằng hai thuật toán Tìm kiếm mù: **BFS (Hàng đợi)** và **DFS (Ngăn xếp)**.

## 📁 Cấu trúc thư mục
- `maps/`: Chứa các bản đồ mê cung (Ma trận 2D).
- `src/core_logic.py`: Chứa các hàm hỗ trợ xác định tọa độ và in bản đồ (Các em không cần sửa file này).
- `src/bfs_solver.py`: Code thuật toán tìm kiếm theo chiều rộng.
- `src/dfs_solver.py`: Code thuật toán tìm kiếm theo chiều sâu.
- `src/main.py`: File thực thi chính.

## 🎯 Nhiệm vụ của các em
1. Mở file `src/bfs_solver.py` và cài đặt logic bằng cấu trúc dữ liệu **Queue** (`collections.deque`).
2. Mở file `src/dfs_solver.py` và cài đặt logic bằng cấu trúc dữ liệu **Stack** (`list`).
3. Thay đổi thứ tự toán tử: Trong hàm get_neighbors, thử đổi thứ tự ưu tiên các hướng di chuyển (ví dụ ưu tiên Phải trước Lên) xem đường đi của DFS có bị thay đổi không.
4. Chạy file `main.py` để xem Robot di chuyển.
5. Tạo một bản đồ mê cung lớn hơn (ví dụ 10x10), có nhiều vật cản. Cho 2 thuật toán chạy và so sánh con số len(visited)
6. Quan sát số lượng ô đã duyệt (Visited Nodes) của BFS và DFS trên bản đồ khó, từ đó rút ra kết luận thuật toán nào tối ưu bộ nhớ hơn.

## 🚀 Cách chạy chương trình
Hãy đảm bảo các em đang đứng ở thư mục gốc của project, mở Terminal và gõ lệnh:
```bash
python src/main.py
