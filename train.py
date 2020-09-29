# import packege
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import Input,Model
# data_pass record
train_pass="/home/silver/img_save/cats_img/train"
test_pass="/home/silver/img_save/cats_img/test"
val_pass="/home/silver/img_save/cats_img/validation"
model_pass="/home/silver/model_save/catvsdog/cvd_model(1).h5"
img_pass="/home/silver/img_save/cats_img/test/cat_test/cat.10.jpg"

# # conv_net define
# model=models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(128,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512,activation='relu'))
# model.add(layers.Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
# model.summary()

input_layer=Input(shape=(150,150,3))
x=layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))(input_layer)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Conv2D(64,(3,3),activation='relu')(x)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Conv2D(128,(3,3),activation='relu')(x)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Conv2D(128,(3,3),activation='relu')(x)
x=layers.MaxPooling2D((2,2))(x)
x=layers.Flatten()(x)
x=layers.Dropout(0.5)(x)
x=layers.Dense(512,activation='relu')(x)
output_layer=layers.Dense(1,activation='sigmoid')(x)

model=Model(input_layer,output_layer)
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
# data pre
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2)
test_datagen=ImageDataGenerator(rescale=1./255)

train_gennerator=train_datagen.flow_from_directory(
    train_pass,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

val_gennerator=test_datagen.flow_from_directory(
    val_pass,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary')

# train data
history=model.fit_generator(
    train_gennerator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=val_gennerator,
    validation_steps=100)

model.save(model_pass)

#model.load_weights(model_pass)
#
# draw a map
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label="train acc")
plt.plot(epochs,val_acc,'b',label="val acc")
plt.title("train && validation's acc")
plt.legend()
plt.figure()

plt.plot(epochs,loss,'bo',label="training loss")
plt.plot(epochs,val_loss,'b',label="val loss")
plt.title("train &&val's loss")
plt.legend()

plt.show()

# get visual image from the net
# img=image.load_img(img_pass,target_size=[150,150])
# img_tensor=image.img_to_array(img)
# img_tensor=np.expand_dims(img_tensor,axis=0)
# img_tensor/=255.
#
#
# layer_out=[layer.output for layer in model.layers[:8]]
# activation_model=models.Model(inputs=model.input,outputs=layer_out)
#
# activations=activation_model.predict(img_tensor)
#
# layer_names=[]
# for layer in model.layers[:8] :
#     layer_names.append(layer.name)
#
# image_per_row=16
# for layer_name,layer_activation in zip(layer_names,activations):
#     n_features=layer_activation.shape[-1]
#     size=layer_activation.shape[1]
#
#     n_cols=n_features//image_per_row
#     display_grid=np.zeros((size*n_cols,image_per_row*size))
#
#     for col in range(n_cols):
#         for row in range(image_per_row):
#             channel_image=layer_activation[0,:,:,col*image_per_row+row]
#             channel_image-=channel_image.mean()
#             channel_image/=channel_image.std()
#             channel_image*=64
#             channel_image+=128
#             channel_image=np.clip(channel_image,0,255).astype('uint8')
#             display_grid[col*size:(col+1)*size,row*size:(row+1)*size]=channel_image
#     scale=1./size
#     plt.figure(figsize=(scale*display_grid.shape[1],scale*display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid,aspect='equal',cmap='viridis')
#
