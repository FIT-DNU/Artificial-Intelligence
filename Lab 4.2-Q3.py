import numpy as np
import matplotlib.pyplot as plt

# Hàm tạo nhiễm sắc thể ngẫu nhiên
def create_chromosome():
    return np.random.randint(0, 4, size=54)

# Hàm tạo quần thể
def create_population(pop_size):
    return np.array([create_chromosome() for _ in range(pop_size)])

# Hàm đánh giá quần thể với yếu tố phạt số bước di chuyển
def evaluate_population(population, room, max_steps=500, penalty_factor=0.1):
    fitness = []
    for chromosome in population:
        efficiency, steps = painter_play(chromosome, room, max_steps)
        if steps >= max_steps:  # Nếu số bước vượt quá giới hạn, áp dụng phạt
            efficiency -= penalty_factor * (steps - max_steps)
        fitness.append(efficiency)
    return np.array(fitness)

# Hàm chọn cha mẹ
def select_parents(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        idx = np.random.choice(np.arange(len(population)), p=fitness / fitness.sum())
        parents[i, :] = population[idx, :]
    return parents

# Hàm lai ghép
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.random.randint(1, offspring_size[1])
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Hàm đột biến
def mutate(offspring, mutation_rate):
    for idx in range(offspring.shape[0]):
        for gene in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[idx, gene] = np.random.randint(0, 4)
    return offspring

# Hàm mô phỏng robot vẽ với số bước
def painter_play(chromosome, room, max_steps=500):
    rows, cols = room.shape
    x, y = np.random.randint(0, rows), np.random.randint(0, cols)
    direction = np.random.randint(0, 4)  # 0: lên, 1: phải, 2: xuống, 3: trái
    painted = np.zeros_like(room)
    steps = 0

    while steps < max_steps:
        if room[x, y] == 0:  # Nếu không phải vật cản
            painted[x, y] = 1  # Sơn ô
        # Xác định trạng thái hiện tại
        c = int(painted[x, y])  # Trạng thái ô hiện tại (0 hoặc 1)
        f = int(room[(x - 1) % rows, y] if direction == 0 else
                room[x, (y + 1) % cols] if direction == 1 else
                room[(x + 1) % rows, y] if direction == 2 else
                room[x, (y - 1) % cols])
        f = 1 if f == 1 else 0  # Đảm bảo f là 0 hoặc 1
        l = int(room[x, (y - 1) % cols] if direction == 0 else
                room[(x - 1) % rows, y] if direction == 1 else
                room[x, (y + 1) % cols] if direction == 2 else
                room[(x + 1) % rows, y])
        l = 1 if l == 1 else 0  # Đảm bảo l là 0 hoặc 1
        r = int(room[x, (y + 1) % cols] if direction == 0 else
                room[(x + 1) % rows, y] if direction == 1 else
                room[x, (y - 1) % cols] if direction == 2 else
                room[(x - 1) % rows, y])
        r = 1 if r == 1 else 0  # Đảm bảo r là 0 hoặc 1
        # Tính chỉ số trạng thái và giới hạn trong phạm vi 0-53
        state = int((c * 27 + f * 9 + l * 3 + r) % 54)
        # Lấy hành động từ nhiễm sắc thể
        action = chromosome[state]
        # Thực hiện hành động
        if action == 1:  # Quay trái
            direction = (direction - 1) % 4
        elif action == 2:  # Quay phải
            direction = (direction + 1) % 4
        elif action == 3:  # Quay ngẫu nhiên
            direction = np.random.randint(0, 4)
        # Di chuyển
        if direction == 0:
            new_x, new_y = (x - 1) % rows, y
        elif direction == 1:
            new_x, new_y = x, (y + 1) % cols
        elif direction == 2:
            new_x, new_y = (x + 1) % rows, y
        elif direction == 3:
            new_x, new_y = x, (y - 1) % cols

        # Nếu không phải vật cản, thực hiện di chuyển
        if room[new_x, new_y] != 2:  # 2 là vật cản
            x, y = new_x, new_y

        steps += 1
        if np.sum(painted) == np.sum(room == 0):  # Nếu đã sơn hết phòng
            break

    efficiency = np.sum(painted) / np.sum(room == 0)  # Tính hiệu quả sơn
    return efficiency, steps

# Thuật toán di truyền
def genetic_algorithm(room, pop_size=50, num_generations=200, mutation_rate=0.002):
    population = create_population(pop_size)
    best_fitness = []
    for generation in range(num_generations):
        fitness = evaluate_population(population, room)
        best_fitness.append(np.max(fitness))
        parents = select_parents(population, fitness, pop_size // 2)
        offspring = crossover(parents, (pop_size - parents.shape[0], population.shape[1]))
        offspring = mutate(offspring, mutation_rate)
        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring
    return population, best_fitness

# Tạo phòng với vật cản ngẫu nhiên
def create_room_with_obstacles(rows, cols, num_obstacles):
    room = np.zeros((rows, cols))
    for _ in range(num_obstacles):
        x, y = np.random.randint(0, rows), np.random.randint(0, cols)
        room[x, y] = 2  # Đánh dấu ô có vật cản
    return room

# Tạo phòng với kích thước 20x40 và thêm 50 vật cản
room = create_room_with_obstacles(20, 40, 50)

# Chạy thuật toán di truyền
population, best_fitness = genetic_algorithm(room)

# Vẽ đồ thị fitness qua các thế hệ
plt.plot(best_fitness)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness vs Generation')
plt.show()

# Kiểm tra nhiễm sắc thể tốt nhất
best_chromosome = population[np.argmax(evaluate_population(population, room))]
efficiency, steps = painter_play(best_chromosome, room)

print(f"Best chromosome efficiency: {efficiency}")
print(f"Steps taken: {steps}")

# In ra phòng sau khi vẽ (với màu sắc)
# Màu: 0 - trắng (chưa sơn), 1 - đen (đã sơn), 2 - xanh lá (vật cản)
# Đối với vật cản (2), sử dụng màu xanh lá; đối với khu vực đã sơn (1), sử dụng màu đen, chưa sơn (0) dùng màu xám nhạt

# Tạo một ảnh với màu sắc rõ ràng để dễ nhận diện
painted_colored = np.zeros((room.shape[0], room.shape[1], 3), dtype=np.uint8)

# Gán màu cho từng khu vực
painted_colored[room == 2] = [0, 255, 0]  # Vật cản (màu xanh lá)
painted_colored[painted == 1] = [0, 0, 0]  # Khu vực đã sơn (màu đen)
painted_colored[room == 0] = [192, 192, 192]  # Khu vực chưa sơn (màu xám nhạt)

# Hiển thị phòng sau khi vẽ
plt.imshow(painted_colored)
plt.title('Painted Room with Obstacles')
plt.show()

