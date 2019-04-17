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
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from keras.layers import BatchNormalization as BN
#from preprocess_CT_image import load_scan, get_pixels_hu, write_dicom, map_0_1


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
        label = np.array(hf.get('label'))
        
        return data, label


# load data. The images are divides to overlapping patcches with size
# and stride defined in train_patches


#data,labels = read_hdf5('Data/Dicom_0/train_40_20_lung_2E3_dicom_0.h5')
## divison by 4095 keeps the input output between 0-1
#data = (data[:,:,:,None]/4095).astype(np.float32)
#labels = (labels[:,:,:,None]/4095).astype(np.float32)

#res = np.subtract(labels,data)



batch_size=256

inputs = Input(shape=(None,None,1))

conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)

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

conv6 = Conv2D(64, (3, 3), padding='same',dilation_rate=(2,2))(conv5)
conv6 = BN()(conv6)
conv6 = Activation('relu')(conv6)

outputs = Conv2D(1, (3, 3), padding='same')(conv6)

model = Model(inputs=[inputs], outputs=[outputs])


    
#model.summary()



#model.load_weights('Weights/weights_prior_pig.h5')
##
#adam=Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=adam,loss=losses.mean_squared_error)
#hist_adam = model.fit(x=data,y=res ,batch_size=batch_size,epochs=20
#        ,validation_split=0.1,verbose=1, shuffle=True)
##model.load_weights('Weights/Lung/weights_prior_lung_2E3.h5')
#
#adam=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(optimizer=adam,loss=losses.mean_squared_error)
#hist_adam = model.fit(x=data,y=res ,batch_size=batch_size,epochs=20
#        ,validation_split=0.1,verbose=1, shuffle=True)
#model.save_weights('Weights/Piglet/weights_prior_adam2_lung_2E3.h5')
#model.save('prior_pig.h5')
model.load_weights('Thoracic/weights_prior_th.h5')

#data_test,labels_test = read_hdf5('Data/test_slice.h5')
#data_test,labels_test = read_hdf5('Data/test_pig_SE.h5')
data_test,labels_test = read_hdf5('Thoracic/test_th_dicom_0.h5')
data_test = (data_test[:,:,:,None]/4095).astype(np.float32)
labels_test = (labels_test[:,:,:,None]/4095).astype(np.float32)

labels_pred = model.predict(data_test,batch_size=1,verbose=1)
labels_pred1 = labels_pred+data_test
# calculate PSNR
labels_pred=(labels_pred1-labels_pred1.min())/(labels_pred1.max() - labels_pred1.min())

diff = labels_test-labels_pred
diff = diff.flatten('C')
rmse = math.sqrt( np.mean(diff ** 2.) )
psnr_prior = 20*math.log10(1.0/rmse)

diff = labels_test-data_test
diff = diff.flatten('C')
rmse = math.sqrt( np.mean(diff ** 2.) )
psnr_test = 20*math.log10(1.0/rmse)



from skimage.measure import compare_ssim 
ssim = 0
for i in range (labels_test.shape[0]):
    ssim = ssim+compare_ssim(labels_test[i,:,:,0], labels_pred[i,:,:,0],
                data_range=labels_pred[i,:,:,0].max() - labels_pred[i,:,:,0].min())
    
ssim_prior = ssim/labels_test.shape[0]
#show one test results
plt.imshow(data_test[10,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow((labels_pred)[10,:,:,0], cmap='gray')
plt.show()
plt.figure()
plt.imshow(labels_test[10,:,:,0], cmap='gray')
plt.show()

#import dicom
#ref = dicom.read_file('C://Users/mansari/Desktop/Dataset/Piglet/Prediction/ref')
#write_dicom(ref ,labels_pred*4095+1024,
#            'C://Users/mansari/Desktop/Dataset/Piglet/Prediction/Prior')

