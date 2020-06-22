
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications.resnet50 import ResNet50

from keras.layers import GlobalAveragePooling2D
from keras import layers
from keras.datasets import cifar10
import numpy as np

"""## 1)  FOR FIXED FEATURE EXTRACTOR"""

# load model without classifier layers
model = ResNet50(include_top=False, input_shape=(224, 224, 3))
#Freezing the layers
for layer in model.layers:
    layer.trainable = False
# add new classifier layers
flat1 = Flatten()(model.outputs)
#class1 = Dense(1024, activation='relu')(flat1)
output = Dense(5, activation='softmax')(flat1)
# define new model
model = Model(inputs=model.inputs, outputs=output)

model.summary()

"""##  2) LAST LAYER TRAINED ON CIFAR 10"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

indexes=[]
index=0
for i,j in zip(x_train,y_train):
    if(j>=5):
        indexes.append(index)
    index=index+1

x_train=np.delete(x_train,indexes,axis=0)
y_train=np.delete(y_train,indexes,axis=0)

x_train=np.delete(x_train,[i for i in range(8000,25000)],axis=0)
y_train=np.delete(y_train,[i for i in range(8000,25000)],axis=0)

num_classes = 5
y_train = keras.utils.to_categorical(y_train, num_classes)

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in conv_base.layers:
    layer.trainable = False

model = Sequential()
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(layers.UpSampling2D((2,2)))
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(5, activation='softmax'))


model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=5, batch_size=20)

model.summary()

