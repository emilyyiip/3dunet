from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from simple_unet_model import simple_unet_model
import os, glob
import nibabel as nib
import keras
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from skimage import io
from patchify import patchify
from keras.layers import Cropping3D
from keras.layers import Cropping3D, ZeroPadding3D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

dataInputPath = '/Users/emilyyip/Desktop/AI_learning/MRI/Anatomical_mag_echo5/'
dataMaskPath = '/Users/emilyyip/Desktop/AI_learning/MRI/whole_liver_segmentation/'
#img_liver = nib.load(dataInputPath)

# STEP 1 - Load and visualize data

imagePathInput = os.path.join(dataInputPath, 'img/')
maskPathInput = os.path.join(dataInputPath, 'mask/')

dataOutputPath = dataInputPath + 'slices'
maskOutputPath = dataMaskPath + 'slices'
imageSliceOutput = os.path.join(dataOutputPath, 'img/')
maskSliceOutput = os.path.join(maskOutputPath, 'mask/')

image_directory = '/Users/emilyyip/Desktop/AI_learning/MRI/Anatomical_mag_echo5/img'
mask_directory = '/Users/emilyyip/Desktop/AI_learning/MRI/whole_liver_segmentation/'

SIZE = 128  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset= []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.
image_dataset = []

images = os.listdir(image_directory)
masks = os.listdir(mask_directory)

#resizing images
for i in os.listdir(image_directory):
    if i.endswith('.nii'):
        image_array = nib.load(os.path.join(image_directory, i)).get_fdata()
        new_img = resize(image_array,(29,128,128), order=1, preserve_range=True)
        image_dataset.append(new_img) 


#resizing masks
for m in masks:
    if m.endswith('.nii'):
        mask_array = nib.load(os.path.join(mask_directory, m)).get_fdata()
        new_mask = resize(mask_array,(29,128,128), order=1, preserve_range=True)
        mask_dataset.append(new_mask)

n_classes = 4
train_img = np.stack((image_dataset,)*3, axis=-1)
train_mask = np.expand_dims(mask_dataset, axis=4)
image_dataset_rgb = np.array([np.repeat(image[:, :, :, np.newaxis], 3, axis=-1) for image in train_img])
mask_dataset_rgb = np.array([np.repeat(mask[:, :, :, np.newaxis], 3, axis=-1) for mask in train_img])
img_mask_selected = image_dataset_rgb[:,:,:,:,:,0]
mask_dataset_selected = mask_dataset_rgb[:,:,:,:,:,0]

X_train, X_test, Y_train, y_test = train_test_split(img_mask_selected, mask_dataset_rgb, test_size = 0.20, random_state = 42)
print(X_train.shape)

###############################################################
IMG_Y= 128
IMG_X  = 128
IMG_Z = 29
IMG_CHANNELS = 3

#######################################################################

def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding = "same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding = "same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation
def adjust_padding_for_concatenation(y, skip):
    # Calculate padding for depth, height, and width
    depth_diff = y.shape[1] - skip.shape[1]
    height_diff = y.shape[2] - skip.shape[2]
    width_diff = y.shape[3] - skip.shape[3]

    depth_pads = (depth_diff // 2, depth_diff - depth_diff // 2) if depth_diff > 0 else (0, 0)
    height_pads = (height_diff // 2, height_diff - height_diff // 2) if height_diff > 0 else (0, 0)
    width_pads = (width_diff // 2, width_diff - width_diff // 2) if width_diff > 0 else (0, 0)

    # Apply padding as needed
    if depth_diff > 0 or height_diff > 0 or width_diff > 0:
        skip = ZeroPadding3D(padding=(depth_pads, height_pads, width_pads))(skip)
    elif depth_diff < 0 or height_diff < 0 or width_diff < 0:
        y = ZeroPadding3D(padding=((abs(depth_pads[0]), abs(depth_pads[1])), (abs(height_pads[0]), abs(height_pads[1])), (abs(width_pads[0]), abs(width_pads[1]))))(y)
    return y, skip

def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(input)
    x, skip_features = adjust_padding_for_concatenation(x, skip_features)
    print(x.shape, skip_features.shape)
    # Concatenation
    x = concatenate([x, skip_features], axis=-1)
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding = "same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model

my_model = build_unet((29,128,128,3), n_classes=4)

print(my_model.summary())

channels=1

LR = 0.0001
optim = keras.optimizers.Adam(LR)
channels = 1
n_classes = 4

def dice_coefficient(y_true, y_pred):
    print("y_true shape:", K.int_shape(y_true))
    print("y_pred shape:", K.int_shape(y_pred))
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

model = build_unet((IMG_X,IMG_Y,IMG_Z,channels), n_classes=n_classes)
model.compile(optimizer=optim, loss=dice_coefficient_loss, metrics=[dice_coefficient])
history=model.fit(X_train, 
          Y_train,
          batch_size=92, 
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test))


          