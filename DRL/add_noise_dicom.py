# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:07:28 2018

@author: Maryam
based on http://www.quarkquark.com/work/Add_Noise_to_CT_Image_ver2.html
"""
import numpy as np
from preprocess_CT_image import load_scan, get_pixels_hu
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

import h5py
    
    
def add_noise_CT(image,voxelSize,theta , N0, mu_water=0.0171):
    epsilon = 5
    
    #now get images to linear attenuation coef (1/mm)
    # slices[0].KVP gives the voltage and from there we can find water attenuation
    #here, 0.0227 is the attenuation coef (1/mm) of water at 50 keV
    image = mu_water*image/1000 + mu_water
    # I get an error using radon transform that asks the values outside the tube to be zero
    image = (image-image.min())
    '''Obtatain Sinogram of the image using radon function'''

    proj = radon(image, theta=theta, circle=True)*voxelSize
    #plot the projections
    #plt.subplot(3,1,3),plt.imshow(sinogram[0],cmap='gray')
       
    '''calculate the amount of Poisson noise for each ray in each projection'''
    N = N0*np.exp(-proj)
    N_noise = np.random.poisson(N)
    proj_noisy_image = -np.log(N_noise/N0)
        
    '''Correct for infs in sinogram'''
    #since sometimes N_noise is <= 0, projs_noise can be inf! We need to correct for
    # this, I will do so here by finding all inf values and replacing them by 
    # -log(epsilon/N0) where eplison is some small number >0 that reflects the
    # smallest possible detected photon count
    
    idx = np.isinf(proj_noisy_image)
    proj_noisy_image[idx] = -np.log(epsilon/N0)/voxelSize
    
    '''reconstruct ct image'''
    # there are 2 ways from here: 
    # 1- iradon on the noisy projection,
    noisy_image = iradon(proj_noisy_image, theta=theta,circle=True)

    #2- iradon the noise and add it to the original image, to avoid messy edges of image
#    projs_noise = proj_noisy_image - proj
#    noise = iradon(projs_noise, theta=theta,circle=True)/voxelSize
#    noisy_image = image + noise
#    
    
    return noisy_image



path = 'C://Users/mansari/Desktop/Dataset/Piglet/Train_High'
#Read the image
#This assumes the image is <> by <> and has data type = float
slices = load_scan(path)

N0 = 3.2315E2 # controls the level of noise added to the image
voxelSize = slices[0].SliceThickness # in mm

'''read the dataset'''
#the image may need to be transformed back to attenuation coefficient
#first get images to HU unit 
image = get_pixels_hu(slices)


''' Add noise to CT'''
theta = np.linspace(0., 360., max(image[0].shape), endpoint=False)
noisy_image = np.zeros(image.shape)
for i in range(0,image.shape[0]):
    print(i)
    image=image.astype(np.float32)
    noisy_image[i] = add_noise_CT(image[i],voxelSize,theta,N0,mu_water=0.0171)
    noisy_image[i] = (noisy_image[i]-noisy_image[i].min())/(noisy_image[i].max()-noisy_image[i].min())
    image[i] = (image[i]-image[i].min())/(image[i].max()-image[i].min())

#plot the image
plt.figure()
plt.subplot(3,1,1),plt.imshow(image[0],cmap='gray')
plt.subplot(3,1,2),plt.imshow(noisy_image[0],cmap='gray')

def write_hdf5(data, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(np.float32)
    #y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        
        
write_hdf5(image,'C://Users/mansari/Desktop/Dataset/Piglet/Data/labels_piglet.h5')
write_hdf5(noisy_image,'C://Users/mansari/Desktop/Dataset/Piglet/Data/data_piglet.h5')