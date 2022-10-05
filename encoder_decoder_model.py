# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:12:50 2020

@author: carlo
"""

import os

import tensorflow as tf
import numpy as np
from PIL import Image
import time
from matplotlib import cm
import matplotlib.pyplot as plt

# Set the seed for random operations. 
# This let our experiments to be reproducible. 
SEED = 1234
tf.random.set_seed(SEED)  
np.random.seed(SEED)

n_exp = '03'
model_name = 'Segm'
cwd = os.getcwd()
dataset_dir = os.path.join(cwd,'Dataset')    #I already have the BipBip Haricot Development dataset (called 'Dataset') in the current directory

#input image shape
img_h = 512
img_w = 512

#batch size
bs = 4

#number of fold to divide the dataset for K-Fold cross-validation
n_kfold = 5

#####################################################################################################
#I build n_kfold folds where to store validation and train txt files
from sklearn.model_selection import KFold

image_set = os.listdir(dataset_dir + '\\Images')  #directory of the images
dataset_len = len(image_set)                      #lenght of the images dataset
images = []                                       #list of images names without extentions
for img in image_set:
    images.append(img.strip('.jpg'))
val_split = 0.2                                   #percentage of the validation dataset
train_card = round(((1-val_split)*dataset_len))   #number of images in training dataset
val_card = round(val_split*dataset_len)           #number of images in validation dataset

train_lis = []
val_lis = []

train_val_shuffle = True            #this makes the shufflig random
if train_val_shuffle:
    np.random.shuffle(images)                    
kf = KFold(n_splits=n_kfold)        #I apply kfold scikit function
k = 0                               #current fold
 
for train_index, test_index in kf.split(images):                #loop over the fold divisions made by KFold function
    train_lis = [images[i] for i in train_index]                #list of images to go in training dataset
    val_lis = [images[i] for i in test_index]                   #list of images to go in validation dataset
    string = 'Splits' + str(k)                                  #name of the folder to store the txt files
    split_dir = os.path.join(dataset_dir, string)               #directory of the folder to store the txt files
    k = k + 1                                                   #increment current fold
    if not os.path.exists(split_dir):                           #I create folder for the current division made by KFold
        os.makedirs(split_dir)
    with open(split_dir + '\\train.txt', 'w') as f:             #write txt file for training dataset
        f.truncate(0)
        for img in train_lis:
            f.write(img + '\r')
        f.close()
    with open(split_dir + '\\val.txt', 'w') as f:               #write txt file for validation dataset
        f.truncate(0)
        for img in val_lis:
            f.write(img + '\r' )
        f.close()
    
#####################################################################################################
#provided function to read RGB masks
def read_rgb_mask(img_path,resize,dim):
    '''
    img_path: path to the mask file
    Returns the numpy array containing target values
    '''

    mask_img = Image.open(img_path)
    if resize:
      mask_img = mask_img.resize(dim, resample=Image.NEAREST)
    mask_arr = np.array(mask_img)

    new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)

    # Use RGB dictionary in 'RGBtoTarget.txt' to convert RGB to target
    new_mask_arr[np.where(np.all(mask_arr == [216, 124, 18], axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == [255, 255, 255], axis=-1))] = 1
    new_mask_arr[np.where(np.all(mask_arr == [216, 67, 82], axis=-1))] = 2

    return new_mask_arr

#####################################################################################################
#provided class to transform manually the data, it was adaped to work in kfold cross validation
class CustomDataset(tf.keras.utils.Sequence):

    """
    CustomDataset inheriting from tf.keras.utils.Sequence.

    3 main methods:
      - __init__: save dataset params like directory, filenames..
      - __len__: return the total number of samples in the dataset
      - __getitem__: return a sample from the dataset

    Note: 
      - the custom dataset return a single sample from the dataset. Then, we use 
        a tf.data.Dataset object to group samples into batches.
      - in this case we have a different structure of the dataset in memory. 
        We have all the images in the same folder and the training and validation splits
        are defined in text files.

    """
    #the parameter num_kfold is added to say in which k_fold I am working with
    def __init__(self, num_kfold, dataset_dir, which_subset, img_generator=None, mask_generator=None, 
               preprocessing_function=None, out_shape=[img_w, img_h]):
        if which_subset == 'training':
            subset_file = os.path.join(dataset_dir, 'Splits' + str(num_kfold), 'train.txt')
        elif which_subset == 'validation':
            subset_file = os.path.join(dataset_dir, 'Splits' + str(num_kfold), 'val.txt')

        with open(subset_file, 'r') as f:
            lines = f.readlines()

        subset_filenames = []
        for line in lines:
            subset_filenames.append(line.strip()) 

        self.which_subset = which_subset
        self.dataset_dir = dataset_dir
        self.subset_filenames = subset_filenames
        self.img_generator = img_generator
        self.mask_generator = mask_generator
        self.preprocessing_function = preprocessing_function
        self.out_shape = out_shape

    def __len__(self):
        return len(self.subset_filenames)

    def __getitem__(self, index):
        # Read Image
        curr_filename = self.subset_filenames[index]
        img = Image.open(os.path.join(self.dataset_dir, 'Images', curr_filename + '.jpg'))
        #mask = Image.open(os.path.join(self.dataset_dir, 'Masks', curr_filename + '.png'))
        

        # Resize image and mask
        img = img.resize(self.out_shape)
        #mask = mask.resize(self.out_shape, resample=Image.NEAREST)

        mask = read_rgb_mask(os.path.join(self.dataset_dir, 'Masks', curr_filename + '.png') , True,self.out_shape) 
        img_arr = np.array(img)
        mask_arr = np.array(mask)

        mask_arr = np.expand_dims(mask_arr, -1)
        out_mask = mask_arr

        if self.which_subset == 'training':
            if ((self.img_generator is not None) and (self.mask_generator is not None)):
                # Perform data augmentation
                # We can get a random transformation from the ImageDataGenerator using get_random_transform
                # and we can apply it to the image using apply_transform
                img_t = self.img_generator.get_random_transform(img_arr.shape, seed=SEED)
                mask_t = self.mask_generator.get_random_transform(mask_arr.shape, seed=SEED)
                img_arr = self.img_generator.apply_transform(img_arr, img_t)
                # ImageDataGenerator use bilinear interpolation for augmenting the images.
                # Thus, when applied to the masks it will output 'interpolated classes', which
                # is an unwanted behaviour. As a trick, we can transform each class mask 
                # separately and then we can cast to integer values (as in the binary segmentation notebook).
                # Finally, we merge the augmented binary masks to obtain the final segmentation mask.
                out_mask = np.zeros_like(mask_arr)
                for c in np.unique(mask_arr):
                    if c > 0:
                        curr_class_arr = np.float32(mask_arr == c)
                        curr_class_arr = self.mask_generator.apply_transform(curr_class_arr, mask_t)
                        # from [0, 1] to {0, 1}
                        curr_class_arr = np.uint8(curr_class_arr)
                        # recover original class
                        curr_class_arr = curr_class_arr * c 
                        out_mask += curr_class_arr
        else:
            out_mask = mask_arr

        if self.preprocessing_function is not None:
            img_arr = self.preprocessing_function(img_arr)

        return img_arr, np.float32(out_mask)

####################################################################################################
# Create training ImageDataGenerator object as usual with data augmentation
# We need two different generators for images and corresponding masks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
apply_data_augmentation = True
if apply_data_augmentation:
    img_data_gen = ImageDataGenerator(rotation_range=90,
                                      width_shift_range=15,
                                      height_shift_range=15,
                                      zoom_range=0.3,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      fill_mode='reflect',
                                      rescale=1./255)
    mask_data_gen = ImageDataGenerator(rotation_range=90,
                                       width_shift_range=15,
                                       height_shift_range=15,
                                       zoom_range=0.3,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='reflect',
                                       rescale=1./255)
else:
    img_data_gen = ImageDataGenerator(rescale=1./255)
    mask_data_gen = ImageDataGenerator(rescale=1./255) 

#####################################################################################################################
#MODEL BUILDER FUNCTION

#downscaling block
def down_block(inpt,filters):
    out1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3),strides=(1, 1), padding='same')(inpt)
    out2 = tf.keras.layers.BatchNormalization()(out1)    #I apply batch normalization before the relu activation
    out3 = tf.keras.layers.ReLU()(out2)
    out4 = tf.keras.layers.MaxPooling2D((2, 2))(out3)
    return out4, out3

#upscaling block
def up_block(inpt,filters,skip_conn):
    out1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), padding='same')(inpt)
    out2 = tf.keras.layers.concatenate([out1, skip_conn])     #I add the skip connection with the correspondent output in the downscaling block
    out3 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same')(out2)
    out4 = tf.keras.layers.BatchNormalization()(out3)    #I apply batch normalization before the relu activation
    out5 = tf.keras.layers.ReLU()(out4)
    return out5

#Model builder
def UnetMod(img_h, img_w, n_class, start_filters, depth):
    inputs = tf.keras.layers.Input([img_h, img_w, n_class])    #input definer
    u = inputs
    l_out = []    #list containing the outputs of the downscaling block to be used in skip connections
    #I add downscaling block with a for loop
    for i in range(0,depth):
        u, u1 = down_block(u, start_filters)
        l_out.append(u1)
        start_filters = start_filters*2
    
    #bottleneck part
    u = tf.keras.layers.Conv2D(start_filters, kernel_size=(3, 3), padding='same') (u)
    u = tf.keras.layers.BatchNormalization()(u)    #I apply batch normalization before the relu activation
    u = tf.keras.layers.ReLU()(u)
    
    
    start_filters = start_filters/2
    
    l_out.reverse()
    #with a for loop I add the upscaling blocks
    for i in range(0,depth-1):
        u = up_block(u,start_filters,l_out[i])
        start_filters = start_filters/2
    
    #final part of the model
    u = tf.keras.layers.Conv2DTranspose(start_filters, kernel_size=(2, 2), strides=(2, 2), padding='same') (u)
    u = tf.keras.layers.concatenate([u, l_out[depth-1]], axis=3)
    u = tf.keras.layers.Conv2D(start_filters, kernel_size=(3, 3), padding='same') (u)
    u = tf.keras.layers.BatchNormalization()(u)    #I apply batch normalization before the relu activation
    u = tf.keras.layers.ReLU()(u)
    
    outputs = tf.keras.layers.Conv2D(filters=n_class, kernel_size=(1, 1), padding='same',activation='softmax') (u)
    
    mod = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return mod
####################################################################################################################       
#I BUILD THE MODEL

model = UnetMod(img_h = img_h, img_w = img_w, n_class = 3, start_filters = 16, depth = 5)

print(model.summary())

# Optimization params
# -------------------

# Loss
# Categoricaly Crossentropy
loss = tf.keras.losses.SparseCategoricalCrossentropy() 

# learning rate
lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
##################################################################################################################
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

####################################################################################################################
#CALLBACKS
#I add the callbacks

#I build the directories
from datetime import datetime

fold_name = 'segmentation_experiments' + n_exp
exps_dir = os.path.join(cwd, fold_name)
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

#I add the model checkpoints
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True)  # False to save the model directly
callbacks.append(ckpt_callback)

#I visualize learning on TensorBoard
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

#I implement early stopping
early_stop = True
pat = 10    #patience of the early stopping
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = pat)
callbacks.append(tb_callback)


#This function allows to see the predicted ouput of the model during training, same code of the lab to plot the images
def show_pred(valid_dataset,epoch=None):
    iterator = iter(valid_dataset)
    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    fig.show()
    image, target = next(iterator)
    index = 3
    image = image[index]
    target = target[index, ..., 0]
    out_sigmoid = model.predict(x=tf.expand_dims(image, 0))

    # Get predicted class as the index corresponding to the maximum value in the vector probability
    # predicted_class = tf.cast(out_sigmoid > score_th, tf.int32)
    # predicted_class = predicted_class[0, ..., 0]
    predicted_class = tf.argmax(out_sigmoid, -1)

    out_sigmoid.shape

    predicted_class = predicted_class[0, ...]

    # Assign colors (just for visualization)
    target_img = np.zeros([target.shape[0], target.shape[1], 3])
    prediction_img = np.zeros([target.shape[0], target.shape[1], 3])
    
    evenly_spaced_interval = np.linspace(0, 1, 2)
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]
    target_img[np.where(target == 0)] = [0, 0, 0]
    for i in range(1, 3):
        target_img[np.where(target == i)] = np.array(colors[i-1])[:3] * 255

    prediction_img[np.where(predicted_class == 0)] = [0, 0, 0]
    for i in range(1, 3):
        prediction_img[np.where(predicted_class == i)] = np.array(colors[i-1])[:3] * 255

    ax[0].imshow(np.uint8(image))
    ax[1].imshow(np.uint8(target_img))
    ax[2].imshow(np.uint8(prediction_img))

    fig.canvas.draw()
    time.sleep(1)


class DispalyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        show_pred(valid_dataset,epoch)
        

callbacks.append(DispalyCallback())

#########################################################################################################
#I TRAIN THE MODEL USING KFOLD CROSS VALIDATION WITH A FOR LOOP 

for i in range(n_kfold):
    
    ##################################################################################################
    #Ibuild the datasets
    
    dataset_train = CustomDataset(i, dataset_dir, 'training', img_generator=img_data_gen, mask_generator=mask_data_gen, preprocessing_function=None)
    dataset_valid = CustomDataset(i, dataset_dir, 'validation', preprocessing_function=None)


    train_dataset = tf.data.Dataset.from_generator(lambda: dataset_train,
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=([img_h, img_w, 3], [img_h, img_w, 1]))

    train_dataset = train_dataset.batch(bs)

    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_generator(lambda: dataset_valid,
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=([img_h, img_w, 3], [img_h, img_w, 1]))
    valid_dataset = valid_dataset.batch(bs)

    valid_dataset = valid_dataset.repeat()

######################################################################################################
    #I train the model with the current KFold division
    epoch_per_kfold = 10
    print("\n" + "Training KFold number " + str(i+1) + " of " + str(n_kfold))
    tf.keras.backend.set_value(model.optimizer.learning_rate, (i+1)*(1e-3))      #to modify manually learning rate during training
    #tf.keras.backend.set_value(model.optimizer.learning_rate, 2*(i+1)*(1e-4))
    hist = model.fit(x=train_dataset,
                     epochs=epoch_per_kfold*(i+1),
                     steps_per_epoch=len(dataset_train),
                     validation_data=valid_dataset,
                     validation_steps=len(dataset_valid),
                     initial_epoch=i*epoch_per_kfold,         #to restart the training in the correct point
                     callbacks=callbacks)
######################################################################################################
    