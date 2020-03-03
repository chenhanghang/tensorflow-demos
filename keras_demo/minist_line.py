import keras # 导入Keras
from keras.datasets import mnist # 从keras中导入mnist数据集
from keras.models import Sequential # 导入序贯模型
from keras.layers import Dense # 导入全连接层
from keras.optimizers import SGD # 导入优化函数
import matplotlib.pyplot as plt # 导入可视化的包

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 下载mnist数据集
print(x_train.shape,y_train.shape) # 60000张28*28的单通道灰度图
print(x_test.shape,y_test.shape)

# im = plt.imshow(x_train[0],cmap='gray')
# plt.show()
# y_train[0]

x_train = x_train.reshape(60000,784) # 将图片摊平，变成向量
x_test = x_test.reshape(10000,784) # 对测试集进行同样的处理

x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

model = Sequential() # 构建一个空的序贯模型
# 添加神经网络层
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=5,validation_data=(x_test,y_test)) # 此处直接将测试集用作了验证集

score = model.evaluate(x_test,y_test)
print("loss:",score[0])
print("accu:",score[1])