import numpy as np
import matplotlib.pyplot as plt

# Hàm để tạo một nhiễm sắc thể ngẫu nhiên
def create_chromosome():
    return np.random.randint(0, 4, size=54)  # Tạo mảng 54 phần tử, giá trị từ 0 đến 3

# Hàm để tạo quần thể ban đầu
def create_population(pop_size):
    return np.array([create_chromosome() for _ in range(pop_size)])

# Hàm để mô phỏng robot sơn phòng
def painter_play(chromosome, room):
    rows, cols = room.shape
    x, y = np.random.randint(0, rows), np.random.randint(0, cols)
    direction = np.random.randint(0, 4)  # 0: up, 1: right, 2: down, 3: left
    painted = np.zeros_like(room)
    steps = 0
    max_steps = 1000  # Giới hạn số bước để tránh vòng lặp vô hạn

    while steps < max_steps:
        if room[x, y] == 0:
            painted[x, y] = 1
        # Xác định trạng thái hiện tại
        c = int(painted[x, y])  # c chỉ có thể là 0 hoặc 1
        f = int(room[(x - 1) % rows, y] if direction == 0 else \
                room[x, (y + 1) % cols] if direction == 1 else \
                room[(x + 1) % rows, y] if direction == 2 else \
                room[x, (y - 1) % cols])
        f = 1 if f == 1 else 0  # Đảm bảo f chỉ là 0 hoặc 1
        l = int(room[x, (y - 1) % cols] if direction == 0 else \
                room[(x - 1) % rows, y] if direction == 1 else \
                room[x, (y + 1) % cols] if direction == 2 else \
                room[(x + 1) % rows, y])
        l = 1 if l == 1 else 0  # Đảm bảo l chỉ là 0 hoặc 1
        r = int(room[x, (y + 1) % cols] if direction == 0 else \
                room[(x + 1) % rows, y] if direction == 1 else \
                room[x, (y - 1) % cols] if direction == 2 else \
                room[(x - 1) % rows, y])
        r = 1 if r == 1 else 0  # Đảm bảo r chỉ là 0 hoặc 1
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
            x = (x - 1) % rows
        elif direction == 1:
            y = (y + 1) % cols
        elif direction == 2:
            x = (x + 1) % rows
        elif direction == 3:
            y = (y - 1) % cols
        steps += 1

    efficiency = np.sum(painted) / np.sum(room == 0)
    return efficiency, painted

# Hàm để đánh giá độ thích nghi của quần thể
def evaluate_population(population, room):
    fitness = []
    for chromosome in population:
        efficiency, _ = painter_play(chromosome, room)
        fitness.append(efficiency)
    return np.array(fitness)

# Hàm để chọn lọc cha mẹ
def select_parents(population, fitness, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        idx = np.random.choice(np.arange(len(population)), p=fitness/fitness.sum())
        parents[i, :] = population[idx, :]
    return parents

# Hàm để lai ghép
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.random.randint(1, offspring_size[1])
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Hàm để đột biến
def mutate(offspring, mutation_rate):
    for idx in range(offspring.shape[0]):
        for gene in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[idx, gene] = np.random.randint(0, 4)
    return offspring

# Hàm chính để chạy thuật toán di truyền
def genetic_algorithm(room, pop_size=50, num_generations=200, mutation_rate=0.002):
    population = create_population(pop_size)
    best_fitness = []
    for generation in range(num_generations):
        fitness = evaluate_population(population, room)
        best_fitness.append(np.max(fitness))
        parents = select_parents(population, fitness, pop_size//2)
        offspring = crossover(parents, (pop_size - parents.shape[0], population.shape[1]))
        offspring = mutate(offspring, mutation_rate)
        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring
    return population, best_fitness

# Tạo phòng trống
room = np.zeros((20, 40))

# Chạy thuật toán di truyền
population, best_fitness = genetic_algorithm(room)

# Vẽ đồ thị độ thích nghi qua các thế hệ
plt.plot(best_fitness)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness vs Generation')
plt.show()

# Kiểm tra nhiễm sắc thể tốt nhất
best_chromosome = population[np.argmax(evaluate_population(population, room))]
efficiency, painted = painter_play(best_chromosome, room)
print(f"Best chromosome efficiency: {efficiency}")

# In ra căn phòng sau khi sơn
plt.imshow(painted, cmap='gray')
plt.title('Painted Room')
plt.show()