# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:23:48 2018

@author: mansari
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:34:57 2017

@author: mansari
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:03:22 2017
Denoising of Dicom image with 3 layers CNN
This project reads Dicom images, stores them in a numpy array and 
applies augmentation. Then it is feed to a 3 layer convolutional 
network with loss = mse

@author: Maryam
"""


import numpy as np

from keras.models import Model
from keras.layers import Dense,concatenate, Activation
from keras.layers import Conv2D, add, Input,Conv2DTranspose
from keras.optimizers import SGD,Adam
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import math
import h5py
from keras.initializers import RandomNormal
from keras import backend as K
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import BatchNormalization as BN
#from DSSIM import dssim

batch_size=32


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        
        return data, label
    
    
def PSNRLoss(y_true, y_pred):

    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

# load data. The images are divides to overlapping patcches with size
# and stride defined in train_patches

#data,labels = read_hdf5('Data/train_slice_35_10.h5')
#data,labels = read_hdf5('Data/train_35_20_pig_SE.h5')
#data,labels = read_hdf5('Data/train_35_10_th.h5')


# 3 layer CNN
inputs = Input(shape=(None,None,1))
conv1 = Conv2D(64, (9,9), activation='relu', padding='same')(inputs)

conv2 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
outputs = Conv2D(1, (3, 3), activation='relu', padding='same' )(conv2)

model = Model(inputs=[inputs], outputs=[outputs])


#model.summary()


ADAM=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd=SGD(lr=0.0001, momentum=0.9, decay=0.9,nesterov=False)

model.compile(optimizer=ADAM,loss=losses.mean_squared_error,metrics=[PSNRLoss])


#hist = model.fit(x=data,y=labels ,batch_size=batch_size,epochs=30
#        ,validation_split=0.2,                              
#         verbose=1, shuffle=True)
#model.save_weights('Thoracic/weights_chen_th.h5')

model.load_weights('Thoracic/weights_chen_th.h5')

#data_test,labels_test = read_hdf5('Data/test_slice.h5')
#data_test,labels_test = read_hdf5('Data/test_pig_SE.h5')
data_test,labels_test = read_hdf5('Thoracic/test_th_dicom_0.h5')
data_test = (data_test[:,:,:,None]/4095).astype(np.float32)
labels_test = (labels_test[:,:,:,None]/4095).astype(np.float32)

labels_pred = model.predict(data_test,batch_size=16,verbose=1)
labels_pred=(labels_pred-labels_pred.min())/(labels_pred.max() - labels_pred.min())
# calculate PSNR
diff = labels_test-labels_pred
diff = diff.flatten('C')
rmse = math.sqrt( np.mean(diff ** 2.) )
psnr = 20*math.log10(1.0/rmse)


# Calculate SSIM
from skimage.measure import compare_ssim 
ssim = 0
for i in range (labels_test.shape[0]):
    ssim = ssim+compare_ssim(labels_test[i,:,:,0], labels_pred[i,:,:,0],
                data_range=labels_pred[i,:,:,0].max() - labels_pred[i,:,:,0].min())
    
ssim = ssim/labels_test.shape[0]

#show one test results
plt.figure()
plt.imshow(data_test[80,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow(labels_pred[80,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow(labels_test[80,:,:,0], cmap='gray')
plt.show()
