import pandas as pd # робота з csv табицями
import numpy as np # математичні операції
import tensorflow as tf # створює нейронку
from tensorflow import keras # розширення до тенсорфлоу
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder # текстові мітки в числа перетворює
import matplotlib.pyplot as plt # графіки

#2 читаємо csv файл
df = pd.read_csv('data/figures_extended.csv')
# print(df.head())

#3 Перетворюємо назви об'єктів (фігур) на числа
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label']) # додаємо один стовпець з числовим значенням назви фігури

#4 Вибираємо стовпці для навчання
X = df[["area", "perimeter", "corners"]]
y = df["label_enc"]

#5 Створюємо модельку для навчання
model = keras.Sequential([
    layers.Dense(8, activation = "relu", input_shape = (3,)), # перший шар - 8 нейронів
    layers.Dense(8, activation = "relu"), # другий шар
    layers.Dense(8, activation = "softmax"), # третій шар
])

#6 Компіляція моделі - визначаємо, як мережа буде навчатися (adam - який краще алгоритм використати для навчання, підбирає ваги, metric - точність)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

#7 Навчання (verbose - після кожної епохи не виводить багато інформації - не навантажує комп)
history = model.fit(X, y, epochs = 300, verbose = 0)

#8 Візуалізація навчання
plt.plot(history.history['loss'], label = "Втрати")
plt.plot(history.history['accuracy'], label = "Точність")
plt.xlabel("Епоха")
plt.ylabel("Значення")
plt.title("Процес навчання моделі")
plt.legend()
plt.savefig('learning_progress.png')
plt.show()

#9 Тестування
test = np.array([[25, 20, 0]])
pred = model.predict(test)
print(f'імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax(pred)])}')