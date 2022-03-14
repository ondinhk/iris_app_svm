import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import svm

dataFrame = pd.read_csv('./iris_dataset.csv')

# Thuộc tính
df = dataFrame.iloc[:, 0:4]
# Nhãn
y = dataFrame.variety

# Tách tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=100)

# Train
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

# Tính độ chính xác
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
      model.score(X_test, y_test))
# Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: 0.978

# Lưu model đã train
# file = open('./model_svm', 'wb')
# pickle.dump(model, file)
#
# Test dự đoán
# Đầu vào là một mảng các giá trị
#
# test = [[7, 3.2, 4.7, 1.4]]
# loaded_model = pickle.load(open('model_svm', 'rb'))
# result = loaded_model.predict(test)
# print(result)
