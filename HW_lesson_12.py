import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.python.layers.normalization import normalization

#1 завантажуємо файли
# Цей метод заходить у папку data/train,
# автоматично читає назви папок (cars, cats, dogs),
# і кожному файлу всередині присвоює відповідну мітку класу.
train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = "categorical")
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                               image_size = (128, 128),
                                                               batch_size = 30,
                                                               label_mode = "categorical")
#2 Нормалізація зображень
normalization_layer = layers.Rescaling(1./255)# переведення в нолики,одинички
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequential()
model.add(layers.Conv2D( # прості ознаки
    filters = 32,  # кількість фільтрів
    kernel_size = (3, 3),  # розмір фільтра
    activation = 'relu',  # функція активації
    input_shape = (128, 128, 3)  # форма вхідного зображення (RGB)
))
model.add(layers.MaxPool2D((2, 2))) # зменшуємо карту ознак у 2 рази

#глибші ознаки (контури, структури)
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D(2,2))

# найскладніші ознаки (форми об'єктів)
model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(3, activation = "softmax")) # 3 класи: cats, dogs, cars

#3 Компіляція моделі
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

#4 Навчання моделі
history = model.fit(train_ds, epochs = 10, validation_data = test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Правдивість: {test_acc}')
class_name = ["apples", "books", "flowers"]

img = image.load_img('images/boo.jpg', target_size = (128, 128))

image_array = image.img_to_array(img)
image_array = image_array / 255.0
image_array = np.expand_dims(image_array, axis = 0)
predictions = model.predict(image_array)
predict_index = np.argmax(predictions[0])

print(f'Імовірність по класах: {predictions[0]}')
print(f'Модель визначила: {class_name[predict_index]}')