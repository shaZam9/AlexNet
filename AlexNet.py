from keras import layers, models, optimizers, activations
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   zca_whitening=True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = True,
                                   fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('./train', target_size=(227, 227), class_mode='categorical')

validation_generator = val_datagen.flow_from_directory('./val', target_size=(227, 227), class_mode='categorical')

#AlexNet creation
model = models.Sequential()
model.add(Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11), strides=4, padding='valid'))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))

model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(Dropout(0.5))

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation(activations.relu))

model.add(Dense(8, activation='softmax'))

model.summary()

def scheduler(epoch):
  if epoch < 10:
    return 0.01
  elif epoch >= 10 and epoch < 15:
    return 0.005
  elif epoch >= 15 and epoch < 20:
    return 0.001
  elif epoch >= 20 and epoch < 25:
    return 0.0005
  else:
    return 0.0001

callback = LearningRateScheduler(scheduler)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.sgd(lr=0.01, momentum=0.9), metrics=['accuracy'])
model.fit_generator(train_generator, epochs=50, validation_data=validation_generator, callbacks=[callback])
