# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:25:08 2018
This code uses perceptual loss for optimization. also shows the metric psnr
loss: vgg16 ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
@author: mansari
"""

import numpy as np

from keras.models import Model
from keras.layers import Dense,concatenate, Activation, Lambda
from keras.layers import Conv2D, add, Input,Conv2DTranspose
from keras.optimizers import SGD,Adam
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import math
import h5py
from keras.initializers import RandomNormal
import tensorflow as tf
from sobel_edge import sobel 
from preprocess_CT_image import load_scan, get_pixels_hu, write_dicom, map_0_1,windowing2
from keras.layers import BatchNormalization as BN

from keras import backend as K
def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels
    
    

def Sobel(x):
    dims = tf.shape (x)#return tf.squeeze( tf.image.sobel_edges(x))
    return tf.reshape(sobel(x),[dims[0],dims[1],dims[2],4])

def Sobel_shape(input_shape):
    dims = [input_shape[0],input_shape[1] ,input_shape[2] ,4]
    output_shape = tuple(dims)
    return output_shape

batch_size=32

# load data. The images are divides to overlapping patcches with size
# and stride defined in train_patches
data,labels = read_hdf5('Data/Dicom/train_128_64_th_dicom_0.h5')

data,labels = read_hdf5('Data/Dicom_0/train_40_20_pig_dicom_0.h5')
# divison by 4095 keeps the input output between 0-1
data = (data[:,:,:,None]/4095).astype(np.float32)
labels = (labels[:,:,:,None]/4095).astype(np.float32)
labels_3 = np.concatenate((labels,labels,labels),axis=-1)
inputs = Input(shape=(None,None,1))

edges = Lambda(Sobel, output_shape = Sobel_shape, name='sobel-edge')(inputs)

input_edge = concatenate([inputs,edges],axis = 3)

conv1 = Conv2D(64, (5,5), activation='relu', padding='same')(input_edge)

conv2 = Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv1)
conv2 = BN()(conv2)
conv2 = Activation('relu')(conv2)

conv3 = Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv2)
conv3 = BN()(conv3)
conv3 = Activation('relu')(conv3)

conv4 = Conv2D(64, (3, 3), padding='same',dilation_rate=(4,4))(conv3)
conv4 = BN()(conv4)
conv4 = Activation('relu')(conv4)

conv5 = Conv2D(64, (3, 3), padding='same',dilation_rate=(3,3))(conv4)
conv5 = BN()(conv5)
conv5 = Activation('relu')(conv5)

conv6 = concatenate([conv5,conv2],axis=3)
conv6 = Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv6)
conv6 = BN()(conv6)
conv6 = Activation('relu')(conv6)

conv7= concatenate([conv6,conv1],axis=3)
conv7 = Conv2D(1, (3, 3), padding='same')(conv7)

conv8= concatenate([conv7,input_edge],axis=3)
outputs = Conv2D(3, (3, 3), padding='same')(conv8)

model_edge_p = Model(inputs=[inputs], outputs=[outputs])
#model_edge_p.summary()


from keras.applications.vgg16 import VGG16
from keras.models import Model
image_shape = (None,None, 3)

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    selectedLayers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
    selectedOutputs = [vgg.get_layer(i).output for i in selectedLayers]
    loss_model = Model(inputs=vgg.input, outputs=selectedOutputs)
    loss_model.trainable = False
    mse = K.variable(value=0)
    for i in range(0,3):
        mse = mse+ K.mean(K.square(loss_model(y_true)[i] - loss_model(y_pred)[i]))
    return mse

model_edge_p.load_weights('Weights/weights_DRL_sobel4d_adam1_perceptual_th.h5')

   
ADAM=Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
model_edge_p.compile(optimizer=ADAM,loss=perceptual_loss,metrics=[PSNRLoss])
hist_adam = model_edge_p.fit(x=data,y=labels_3,batch_size=batch_size,epochs=20
                     ,validation_split=0, verbose=1, shuffle=True)
model_edge_p.save_weights('Weights/weights_DRL_edge4d_adam1_perceptual_pig.h5')

ADAM=Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
model_edge_p.compile(optimizer=ADAM,loss=perceptual_loss,metrics=[PSNRLoss])
hist_adam = model_edge_p.fit(x=data,y=labels_3,batch_size=batch_size,epochs=20
                     ,validation_split=0, verbose=1, shuffle=True)
model_edge_p.save_weights('Weights/weights_DRL_edge4d_adam2_perceptual_pig.h5')

model_edge_p.save('DRL_edge_p.h5')
###Test
model_edge_p.load_weights('Weights/weights_DRL_edge4d_adam2_perceptual_pig.h5')

data_test,labels_test = read_hdf5('Data/Dicom_0/test_pig_dicom_0.h5')
data_test = (data_test[:,:,:,None]/4095).astype(np.float32)
labels_test = (labels_test[:,:,:,None]/4095).astype(np.float32)
labels_pred = model_edge_p.predict(data_test,batch_size=8,verbose=1)

# calculate PSNR
diff = labels_test-labels_pred
diff = diff.flatten('C')
rmse = math.sqrt( np.mean(diff ** 2.) )
psnr_edge_p = 20*math.log10(1.0/rmse)
#


# Calculate SSIM
# Calculate SSIM
from skimage.measure import compare_ssim 
ssim = 0
for i in range (labels_test.shape[0]):
    ssim = ssim+compare_ssim(labels_test[i,:,:,0], labels_pred[i,:,:,0],
                data_range=labels_pred[i,:,:,0].max() - labels_pred[i,:,:,0].min())
    
ssim_edge_p = ssim/labels_test.shape[0]

labels_test_3 = np.concatenate((labels_test,labels_test,labels_test),axis=-1)

[perceptual_loss_edge_p,psnr]=model_edge_p.evaluate(x=data_test,y=labels_test_3, batch_size=8,verbose=1)
#w_labels=windowing2((labels_test)*4095+1024,40,400)
#w_data=windowing2((data_test)*4095+1024,40,400)
#w_pred=windowing2((labels_pred)*4095+1024,40,400)
##show one test results
plt.imshow(data_test[100,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow((labels_pred)[100,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow(labels_test[100,:,:,0], cmap='gray')
plt.show()

import dicom
ref = dicom.read_file('C://Users/mansari/Desktop/Dataset/Piglet/Prediction/ref.dcm') 
#write_dicom(ref ,data_test*4095+1024,
#            'C://Users/mansari/Desktop/Dataset/Lung/Test_Low')
#write_dicom(ref ,labels_test*4095+1024,
#            'C://Users/mansari/Desktop/Dataset/Lung/Test_High')
write_dicom(ref ,labels_pred*4095+1024,
            'C://Users/mansari/Desktop/Dataset/Piglet/Prediction/DRL-Edge-Perceptual')