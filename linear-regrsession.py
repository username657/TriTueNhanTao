import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình trên dữ liệu
model.fit(x.reshape(-1, 1), y)  # Reshape x để tương thích

# In các hệ số (độ dốc và giao điểm)
print("Hệ số:", model.coef_, model.intercept_)

# Dự đoán giá trị y cho các giá trị x mới
x_new = np.array([0, 6, 10])  # Ví dụ về các giá trị x mới
predicted_y = model.predict(x_new.reshape(-1, 1))

# Hiển thị đường hồi quy và các điểm dữ liệu
plt.plot(x, y, 'o', label='Dữ liệu')
plt.plot(x_new, predicted_y, label='Dự đoán')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hồi quy tuyến tính')
plt.legend()
plt.show()