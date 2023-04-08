from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

from keras import utils
import matplotlib.pyplot as plt
import numpy as np


#Загрузка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#названия классов
classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

plt.figure(figsize=(10,10))
for i in range(100,150):
    plt.subplot(5,10,i-100+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    #plt

#Преобразование размерности изображений
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
input_shape = (28, 28, 1)

#Нормализация данных
x_train = x_train / 255
x_test = x_test / 255

y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test,10)


#Последовательная модель, в которую добавляем уровни сети(полносвязные слои)
model = Sequential()
model.add(Conv2D(75, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(100, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#Компилирую модель
model.compile(loss="categorical_crossentropy", optimizer="Nadam", metrics=["accuracy"])

#Обучение
model.fit(x_train, y_train,
          batch_size = 200,
          epochs = 10,
          verbose = 1)

#Запускаем сеть на входных данных
predictions = model.predict(x_test)

print (classes)

for n in range(1, 110):
    plt.imshow(x_test[n+1].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    print(n)
    print(predictions[n])

    #Выводим номер класса(индекс максимального значения в массиве)
    print("Определено", np.argmax(predictions[n]), "Ожидалось", np.argmax(y_test[n]))
    print("Определено", classes[np.argmax(predictions[n])], "Ожидалось", classes[np.argmax(y_test[n])])
