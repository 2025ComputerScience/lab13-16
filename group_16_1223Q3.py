import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

# 設定隨機種子
np.random.seed(1234)

# 載入 MNIST 資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 資料預處理
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 建立 MLP 模型
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 評估模型
loss, acc = model.evaluate(X_test, y_test)
print(f"測試準確率：{acc:.4f}")

# 載入你自己畫的數字圖片（例如 digit.png）
img = Image.open("digit.png").convert("L").resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array  # 反轉顏色（白底黑字）
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28)

# 預測
pred = model.predict(img_array)
print("預測結果：", np.argmax(pred))
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"預測：{np.argmax(pred)}")
plt.show()