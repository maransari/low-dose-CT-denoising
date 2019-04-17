Implementation of <a href="https://arxiv.org/abs/1902.10127">Deep Learning for Low-Dose CT Denoising</a> 

low-dose CT denoising with deep learning 

preprocess_CT_image.py includes all the function needed to work with Dicom images. 

add_noise_dicom.py adds Poisson noise to the sinogram data of normal-dose CT images and generates low-dose CT images.

DRL.py refers to the dilated residual learning network published in [1]. The loss function is mean-square error.

DRL_Edge.py adds an edge detection layer in the begining of the DRL network and the loss function is mean-square error.

DRL_Edge_Perceptual similar to DRL_Edge network wih perceptual loss is the objective function.

DRL_Edge_Perceptual_mse.py similar to DRL_Edge network wih the joint objective funtion (perceptual loss and mean square error).




[1] M. Gholizadeh-Ansari, J. Alirezaie, and P. Babyn, “Low-dose CT
denoising with dilated residual network,” in 2018 40th Annual International
Conference of the IEEE Engineering in Medicine and Biology
Society (EMBC). IEEE, 2018, pp. 5117–5120.
