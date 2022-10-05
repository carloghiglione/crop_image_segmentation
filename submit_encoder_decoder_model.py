# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:53:05 2020

@author: carlo
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from matplotlib import cm
import matplotlib.pyplot as plt
import json
import cv2

SEED = 1234
tf.random.set_seed(SEED)  
np.random.seed(SEED)

#the foder Test_Dev must be in the current work directory
curr_dir = os.getcwd()  
n_exp = '03'
fold_name = 'segmentation_experiments' + n_exp
ckpt_dir = os.path.join(curr_dir, fold_name)
folder_used = 'Segm_2020-12-08_11-14-41'
ckpt_dir = os.path.join(ckpt_dir,folder_used)
ckpt_dir = os.path.join(ckpt_dir, 'ckpts')


used_ckpt = 'cp_46' #metti quello di tensorboard + 1
used_ckpt_address = used_ckpt + '.ckpt'

ckpt_elem = os.path.join(ckpt_dir,used_ckpt_address)


img_h = 512
img_w = 512

bs = 4

#####################################################################################
def read_rgb_mask(img_path):
    '''
    img_path: path to the mask file
    Returns the numpy array containing target values
    '''

    mask_img = Image.open(img_path)
    mask_arr = np.array(mask_img)

    new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)

    # Use RGB dictionary in 'RGBtoTarget.txt' to convert RGB to target
    new_mask_arr[np.where(np.all(mask_arr == [216, 124, 18], axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == [255, 255, 255], axis=-1))] = 1
    new_mask_arr[np.where(np.all(mask_arr == [216, 67, 82], axis=-1))] = 2

    return new_mask_arr


#######################################################################################

#I create the model

def down_block(inpt,filters):
    out1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3),strides=(1, 1), padding='same')(inpt)
    out2 = tf.keras.layers.BatchNormalization()(out1)    #I apply batch normalization before the relu activation
    out3 = tf.keras.layers.ReLU()(out2)
    out4 = tf.keras.layers.MaxPooling2D((2, 2))(out3)
    return out4, out3

def up_block(inpt,filters,skip_conn):
    out1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inpt)
    out2 = tf.keras.layers.concatenate([out1, skip_conn])
    out3 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(out2)
    out4 = tf.keras.layers.BatchNormalization()(out3)    #I apply batch normalization before the relu activation
    out5 = tf.keras.layers.ReLU()(out4)
    return out5

def UnetMod(img_h, img_w, n_class, start_filters, depth):
    inputs = tf.keras.layers.Input([img_h, img_w, n_class])
    u = inputs
    l_out = []
    for i in range(0,depth):
        u, u1 = down_block(u, start_filters)
        l_out.append(u1)
        start_filters = start_filters*2
        
    u = tf.keras.layers.Conv2D(start_filters, kernel_size=(3, 3), padding='same') (u)
    u = tf.keras.layers.BatchNormalization()(u)    #I apply batch normalization before the relu activation
    u = tf.keras.layers.ReLU()(u)
    
    
    start_filters = start_filters/2
    
    l_out.reverse()
    for i in range(0,depth-1):
        u = up_block(u,start_filters,l_out[i])
        start_filters = start_filters/2
    u = tf.keras.layers.Conv2DTranspose(start_filters, kernel_size=(2, 2), strides=(2, 2), padding='same') (u)
    u = tf.keras.layers.concatenate([u, l_out[depth-1]], axis=3)
    u = tf.keras.layers.Conv2D(start_filters, kernel_size=(3, 3), padding='same') (u)
    u = tf.keras.layers.BatchNormalization()(u)    #I apply batch normalization before the relu activation
    u = tf.keras.layers.ReLU()(u)
    
    outputs = tf.keras.layers.Conv2D(filters=n_class, kernel_size=(1, 1), padding='same',activation='softmax') (u)
    
    mod = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return mod
        


model = UnetMod(img_h = img_h, img_w = img_w, n_class = 3, start_filters = 16, depth = 5)

print(model.summary())

#############################################################################################
# Optimization params
# -------------------

# Loss
# Categoricaly Crossentropy
loss = tf.keras.losses.SparseCategoricalCrossentropy() 

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


# Here we define the intersection over union for each class in the batch.
# Then we compute the final iou as the mean over classes
def meanIoU(y_true, y_pred):
    # get predicted class from softmax
    y_pred = tf.expand_dims(tf.argmax(y_pred, -1), -1)

    per_class_iou = []

    for i in range(1,3): # exclude the background class 0
      # Get prediction and target related to only a single class (i)
      
      class_pred = tf.cast(tf.where(y_pred == i, 1, 0), tf.float32)
      class_true = tf.cast(tf.where(y_true == i, 1, 0), tf.float32)
      intersection = tf.reduce_sum(class_true * class_pred)
      union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred) - intersection
    
      iou = (intersection + 1e-7) / (union + 1e-7)
      per_class_iou.append(iou)

    return tf.reduce_mean(per_class_iou)

metrics = ['accuracy', meanIoU]
# ------------------

# Compile Model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#I load the weights
model.load_weights(ckpt_elem)

##############################################################################################

#I do the predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_img_data_gen = ImageDataGenerator()

test_dir = os.path.join(curr_dir, 'Test_Dev\\Bipbip\\Haricot')
test_gen = test_img_data_gen.flow_from_directory(test_dir,
                                                 target_size=(img_h, img_w), 
                                                 class_mode=None, 
                                                 shuffle=False,
                                                 interpolation='bilinear',
                                                 seed=SEED,
                                                 classes = ['Images'])

test_dataset = tf.data.Dataset.from_generator(lambda: test_gen,
                                               output_types=tf.float32,
                                               output_shapes=[None, img_h, img_w, 3])

iterator = iter(test_dataset)

out = next(iterator)

#############################################################################################
#I test the model on a picture in the test set
index = 5
image = out[index]

prediction = model.predict(tf.expand_dims(image, 0))

predicted_class = tf.argmax(prediction, -1)

predicted_class = predicted_class[0, ...]

# Assign colors (just for visualization)
prediction_img = np.zeros([img_h, img_w, 3])
prediction_img[np.where(predicted_class == 0)] = [0, 0, 0]
prediction_img[np.where(predicted_class == 1)] = [255, 255, 255]
prediction_img[np.where(predicted_class == 2)] = [255, 0, 0]

plt.imshow(prediction_img)

################################################################################################

def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

###############################################################################################

teams = ["Bipbip", "Pead", "Roseau", "Weedelec"]
crops = ["Haricot", "Mais"]

submission_dict = {}

for team in teams:
    for crop in crops:
        
        #directory
        test_dir = os.path.join(curr_dir, 'Test_Dev', team, crop, 'Images')
        
        #I get the list of image names
        names_list = os.listdir(test_dir)
        names_list = sorted(names_list)
        
        for img_name in names_list:
            
            image = Image.open(os.path.join(test_dir, img_name))
            
            img_name = img_name.replace('.jpg',"").replace('.png',"")
            
            initial_size = list(image.size)
            image = image.resize([img_w, img_h])
            
            img_arr = np.array(image)
            #img_arr = preprocess_input(img_arr)  #If I have used some prepocess like the one for vgg
            
            prediction = model.predict(tf.expand_dims(img_arr, axis=0))
            mask_arr = tf.argmax(prediction, -1) 
            mask_arr = np.array(mask_arr[0, ...])
            
            mask_arr = cv2.resize(mask_arr, dsize=(initial_size[0], initial_size[1]), interpolation=cv2.INTER_NEAREST)
            
            submission_dict[img_name] = {}
            submission_dict[img_name]['shape'] = mask_arr.shape
            submission_dict[img_name]['team'] = team
            submission_dict[img_name]['crop'] = crop
            submission_dict[img_name]['segmentation'] = {}

            rle_encoded_crop = rle_encode(mask_arr == 1)
            rle_encoded_weed = rle_encode(mask_arr == 2)

            submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
            submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed
            
            
            with open('submission.json', 'w') as f:
                json.dump(submission_dict, f)
            