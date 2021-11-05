import numpy as np
import tensorflow.keras.optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import random
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import keras
from tqdm import tqdm
import glob 

file_path = '../input/sports-image-dataset/data' 


train,validation=data_pre_processing(file_path)
observing_the_data(train,validation)
labels_distribution_bar(train)
vgg_model=model()
vgg_model=configure_model(vgg_model)

#************Hyper-Parameters*************

batch_size=32
epochs=100
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.00003)
loss=tensorflow.keras.losses.categorical_crossentropy
validation_steps=validation.samples/batch_size
train_steps=train.samples/batch_size

#************Train the model********************
vgg_model.compile(optimizer=opt,loss=tensorflow.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
history=vgg_model.fit_generator(generator=train,steps_per_epoch=train_steps,epochs=epochs,verbose=1
                                ,callbacks=callbacks(),validation_data=validation,validation_steps=validation_steps,validation_freq=1,class_weight=class_weights1)

"***********Plots the results*************
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["loss","Validation Loss"])
plt.show()


plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Accuracy","Validation Accuracy"])
plt.show()

vgg_model.save('vgg16.h5')

path='../input/d/omreekapon/dataset'

labels=dict()
for label_name,label_num in validation.class_indices.items():
      labels[label_num]=label_name
predictions=[]

data_path = os.path.join(path,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    data.append(img)
resized_images=[]
   
for image in data:
     image=cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
     resized_images.append(image)
     yhat = vgg_model.predict(image)
     predictions.append(labels[np.argmax(yhat)])
resized_images=np.array(resized_images)

plt.figure(figsize=(10,10))
for i in tqdm(range(12)):
    plt.subplot(4,3,i+1)
    plt.tight_layout(h_pad=5)
    img=resized_images[i]
    img = np.squeeze(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.title(predictions[i])
    plt.xticks(())
    plt.yticks(())
    plt.imshow(img)
plt.show()
