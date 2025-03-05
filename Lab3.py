import numpy as np

# Tạo dữ liệu huấn luyện
def create_train_data():
    data = [['Sunny', 'Hot', 'High', 'Weak', 'no'],
            ['Sunny', 'Hot', 'High', 'Strong', 'no'],
            ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'no'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
            ['Overcast', 'Mild', 'High', 'Weak', 'no'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'yes']]
    return np.array(data)

# Tính xác suất tiên nghiệm (Prior Probability)
def compute_prior_probability(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    total_samples = train_data.shape[0]
    
    for i, label in enumerate(y_unique):
        prior_probability[i] = np.sum(train_data[:, -1] == label) / total_samples
    return prior_probability

# Tính xác suất có điều kiện (Conditional Probabilities)
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    
    for i in range(0, train_data.shape[1] - 1):  # Lặp qua tất cả các đặc trưng ngoại trừ cột nhãn
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        feature_conditional_prob = []
        
        for label in y_unique:
            label_data = train_data[train_data[:, -1] == label]
            feature_prob = []
            for x in x_unique:
                feature_prob.append(np.sum(label_data[:, i] == x) / label_data.shape[0])
            feature_conditional_prob.append(feature_prob)
        
        conditional_probability.append(np.array(feature_conditional_prob))
    
    return conditional_probability, list_x_name

# Lấy chỉ số của giá trị đặc trưng
def get_index_from_value(feature_name, list_features):
    return np.where(list_features == feature_name)[0][0]

# Huấn luyện mô hình Naive Bayes
def train_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(train_data)
    return prior_probability, conditional_probability, list_x_name

# Dự đoán việc chơi tennis (mô hình Naive Bayes)
def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):
    # Chuyển các đặc trưng đầu vào thành chỉ số dựa trên danh sách các giá trị duy nhất
    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    # Tính toán xác suất cho mỗi lớp ('no' và 'yes')
    p0 = prior_probability[0]
    p1 = prior_probability[1]
    
    # Nhân các xác suất có điều kiện của mỗi đặc trưng với lớp
    p0 *= np.prod(conditional_probability[0][0, [x1, x2, x3, x4]])
    p1 *= np.prod(conditional_probability[0][1, [x1, x2, x3, x4]])

    # So sánh các xác suất
    if p0 > p1:
        y_pred = 0  # 'no'
    else:
        y_pred = 1  # 'yes'
    
    return y_pred

# Ví dụ dự đoán
X = ['Sunny', 'Cool', 'High', 'Strong']
train_data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(train_data)
pred = prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability)

if pred:
    print("A should go!")
else:
    print("A should not go!")
