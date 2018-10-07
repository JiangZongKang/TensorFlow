import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

nb_class = 10
nb_epoch = 20
batchsize = 1024

# 数据预处理
(x_train, y_trin), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)

# 设定数据的形状
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# one-hot 编码
y_trin = np_utils.to_categorical(y_trin, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

# 设置模型
model = Sequential()

# 1st conv2d layer
model.add(Convolution2D(
    filters=32,
    kernel_size=[5, 5],
    padding='same',
    input_shape=(28, 28, 1)
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))

# 2sd conv2d layer
model.add(Convolution2D(
    filters=64,
    kernel_size=[5, 5],
    padding='same'
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))
# 1st full conneted dense
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# 2sd full conneted dense
model.add(Dense(10))
model.add(Activation('softmax'))

# 定义优化器和其他参数
adam = Adam(lr=0.0001)

# 编译model
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 训练神经网络
model.fit(
    x_train, y_trin,
    batch_size=batchsize,
    epochs=nb_epoch,
    validation_data=(x_test, y_test)
)
evaluation = model.evaluate(x_test, y_test)
print(evaluation)
