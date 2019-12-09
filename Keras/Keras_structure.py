### 以下为Kera模型介绍，本程序无法执行
## 序列模型
from keras.models import Sequential

# Create the Sequential model
# keras.models.Sequential类是神经网络模型的封装容器。
# 它会提供常见的函数，如fit(),evaluate(),compile()
model = Sequential()

## 层
# kera层就像神经网络层。有全连接层、最大池化层和激活层。
# 可以使用add()函数添加层。
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten

# 创建序列模型
model = Sequential()

# 第一层-添加有128个节点的全连接层以及32个节点的输入层
model.add(Dense(128,input_dim=32))

# 第二层-添加softmax激活层
model.add(Activation('softmax'))

# 第三层-添加全连接层
model.add(Dense(10))

# 第四层-添加sigmoid激活层
model.add(Activation('sigmoid'))

# 对模型进行编译
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(X,y,epochs=1000,verbose=0)

model.evaluate()

