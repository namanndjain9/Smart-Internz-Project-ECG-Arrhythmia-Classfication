from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential


train_datagen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True,vertical_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r"C:\Users\Naman\Desktop\Data Collection\data\train",target_size=(64,64),batch_size=32,class_mode="categorical")
x_test=test_datagen.flow_from_directory(r"C:\Users\Naman\Desktop\Data Collection\data\test",target_size=(64,64),batch_size=32,class_mode="categorical")

model=Sequential()
model.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=600,activation='relu'))
model.add(Dense(units=600,activation='relu'))
model.add(Dense(units=600,activation='relu'))
model.add(Dense(units=6,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics='accuracy')

model.fit_generator(x_train, steps_per_epoch=50, epochs=40, validation_data=x_test, validation_steps=20)

model.save('project.h5')
