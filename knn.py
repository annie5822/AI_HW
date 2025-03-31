import csv
import math
import random

random.seed(135)
# 加載 CSV 文件並轉換為列表格式
def load_csv(filename):
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    header = data[0]  # 提取標題行
    return data[1:], header  # 返回數據和標題

# 將分類數據映射到數值
def map_values(data, header, mapping):
    for row in data:
        for col_name, value_map in mapping.items():
            col = header.index(col_name)
            if row[col] in value_map:
                row[col] = value_map[row[col]]
            else:
                try:
                    row[col] = float(row[col])  # 保留數值欄位
                except ValueError:
                    print(f"Warning: Unexpected value '{row[col]}' in column '{col_name}'.")

# 計算均值
def calculate_means(data):
    means = []
    for col in range(len(data[0])):
        col_sum = sum(float(row[col]) for row in data)
        means.append(col_sum / len(data))
    return means

# 計算標準差
def calculate_stds(data, means):
    stds = []
    for col in range(len(data[0])):
        variance = sum((float(row[col]) - means[col]) ** 2 for row in data) / len(data)
        stds.append(math.sqrt(variance))
    return stds

# 標準化數據
def standardize_data(data, means, stds):
    for row in data:
        for col in range(len(row)):
            if col < len(means):
                row[col] = (float(row[col]) - means[col]) / stds[col] if stds[col] != 0 else 0

# 計算協方差矩陣
def calculate_covariance_matrix(data):
    num_rows = len(data)
    num_cols = len(data[0])
    cov_matrix = [[0.0] * num_cols for _ in range(num_cols)]
    
    for i in range(num_cols):
        for j in range(num_cols):
            cov_sum = sum(float(row[i]) * float(row[j]) for row in data)
            cov_matrix[i][j] = cov_sum / (num_rows - 1)
    
    return cov_matrix

# 計算特徵值和特徵向量（簡單替代）
def compute_eigenvalues_eigenvectors(matrix):
    eigvals = [sum(row) for row in matrix]  # 簡單替代
    eigvecs = [[1 if i == j else 0 for i in range(len(matrix))] for j in range(len(matrix))]  # 單位向量替代
    return eigvals, eigvecs

# 將數據投影到主成分空間
def project_data(data, eigenvectors, num_components=1):
    projected_data = []
    for row in data:
        projected_row = [sum(float(row[i]) * eigenvectors[j][i] for i in range(len(row))) for j in range(num_components)]
        projected_data.append(projected_row)
    return projected_data

# 主成分分析（PCA）實現
def pca(data, num_components=1):
    means = calculate_means(data)
    stds = calculate_stds(data, means)
    standardize_data(data, means, stds)
    cov_matrix = calculate_covariance_matrix(data)
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov_matrix)
    
    sorted_indices = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
    selected_eigenvectors = [eigenvectors[i] for i in sorted_indices[:num_components]]
    
    projected_data = project_data(data, selected_eigenvectors, num_components)
    return projected_data

# 歐式距離計算函數
def euclidean_distance(train_row, test_row):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(train_row, test_row)))

# KNN 預測函數
def knn_predict(train_data, train_labels, test_data, k):
    predictions = []
    for test_row_index in range(len(test_data)):
        test_row = test_data[test_row_index]
        distances = []
        for train_row_index in range(len(train_data)):
            train_row = train_data[train_row_index]
            distance = euclidean_distance(train_row, test_row)
            distances.append((distance, train_labels[train_row_index]))
        
        # 選取最近的 k 個鄰居
        k_nearest = sorted(distances, key=lambda x: x[0])[:k]
        
        # 投票統計
        votes = [label for _, label in k_nearest]
        prediction = 1 if sum(votes) > k / 2 else 0
        predictions.append(prediction)
    
    return predictions

# 保存預測結果到 CSV 文件
def save_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Churn"])
        for value in data:
            writer.writerow(["Yes" if value == 1 else "No"])

# K-fold 交叉驗證函數展開
def k_fold_cross_validation(data, labels, num_components_list, k_values, num_folds=5):
    fold_size = len(data) // num_folds
    indices = list(range(len(data)))
    random.shuffle(indices)

    best_accuracy = 0
    best_num_components = None
    best_k = None

    for num_components_index in range(len(num_components_list)):
        num_components = num_components_list[num_components_index]
        for k_index in range(len(k_values)):
            k = k_values[k_index]
            fold_accuracies = []
            
            for i in range(num_folds):
                val_indices = indices[i * fold_size: (i + 1) * fold_size]
                train_indices = [idx for idx in indices if idx not in val_indices]
                
                train_data = [data[idx] for idx in train_indices]
                train_labels = [labels[idx] for idx in train_indices]
                val_data = [data[idx] for idx in val_indices]
                val_labels = [labels[idx] for idx in val_indices]
                
                train_data_pca = pca(train_data, num_components=num_components)
                val_data_pca = pca(val_data, num_components=num_components)
                
                val_predictions = knn_predict(train_data_pca, train_labels, val_data_pca, k=k)
                
                correct = 0
                for val_idx in range(len(val_labels)):
                    if val_predictions[val_idx] == val_labels[val_idx]:
                        correct += 1
                accuracy = correct / len(val_labels)
                fold_accuracies.append(accuracy)
            
            mean_accuracy = 0
            for fold_accuracy in fold_accuracies:
                mean_accuracy += fold_accuracy
            mean_accuracy /= num_folds
            print(f"num_components={num_components}, k={k}, Mean Accuracy={mean_accuracy:.4f}")
            
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_num_components = num_components
                best_k = k

    print(f"\nBest num_components={best_num_components}, Best k={best_k}, Best Mean Accuracy={best_accuracy:.4f}")
    return best_num_components, best_k

# 加載訓練和驗證數據
train_data, header = load_csv('C:/Users/user/Desktop/hw4/train.csv')
churn_data, _ = load_csv('C:/Users/user/Desktop/hw4/train_gt.csv')
val_data, _ = load_csv('C:/Users/user/Desktop/hw4/val.csv')
test_data, _ = load_csv('C:/Users/user/Desktop/hw4/test.csv')

# 定義映射表
binary_mapping = {'gender': {'Female': 0, 'Male': 1},
                  'SeniorCitizen': {'No': 0, 'Yes': 1},
                  'Partner': {'No': 0, 'Yes': 1},
                  'Dependents': {'No': 0, 'Yes': 1},
                  'PhoneService': {'No': 0, 'Yes': 1},
                  'PaperlessBilling': {'No': 0, 'Yes': 1}}

multi_category_mapping = {'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
                          'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
                          'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
                          'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
                          'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 
                                            'Bank transfer (automatic)': 2, 
                                            'Credit card (automatic)': 3}}

# 對數據進行映射
map_values(train_data, header, binary_mapping)
map_values(train_data, header, multi_category_mapping)
map_values(val_data, header, binary_mapping)
map_values(val_data, header, multi_category_mapping)
map_values(test_data, header, binary_mapping)
map_values(test_data, header, multi_category_mapping)


# 提取標籤並刪除原始數據中的標籤
train_labels = [0 if row[0] == 'No' else 1 for row in churn_data]
train_data = [row[1:] for row in train_data]

# 計算均值和標準差，並對數據標準化
means = calculate_means(train_data)
stds = calculate_stds(train_data, means)
standardize_data(train_data, means, stds)
standardize_data(val_data, means, stds)
standardize_data(test_data, means, stds)

# 搜索最佳的 num_components 和 k 值
num_components_list = [3, 4, 5, 6]  # 可以根據需要調整範圍
k_values = [11, 13, 15, 19, 21, 23, 25]  # 可以根據需要調整範圍

best_num_components, best_k = k_fold_cross_validation(train_data, train_labels, num_components_list, k_values, num_folds=5)

# 使用最佳參數在 train_data 上訓練並在 val_data 上預測
train_data_pca = pca(train_data, num_components=best_num_components)
val_data_pca = pca(val_data, num_components=best_num_components)
test_data_pca = pca(test_data, num_components=best_num_components)

val_predictions = knn_predict(train_data_pca, train_labels, val_data_pca, k=best_k)
test_predictions = knn_predict(train_data_pca, train_labels, test_data_pca, k=best_k)

# 保存最終預測結果
save_csv('val_pred.csv', val_predictions)
save_csv('test_pred.csv', test_predictions)
