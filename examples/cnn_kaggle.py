import numpy as np
import os
import pandas as pd

from dlkeras.cnn import CNN

# For the CNN example we will use the kaggle dataset
# from the Statoil competition. In this competition, we need to
# classify ships and icebergs in satellite images.
# The data can be found through the link below:
# https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data

# directory where results will be saved
save_dir = '/.../output/'

# specify train and test data paths
path_train = '/.../input/train.json'
path_test = '/.../input/test.json'

# load data
train_raw = pd.read_json(path_train, precise_float=True)
train_labels = train_raw['is_iceberg'].values
train_ids = train_raw['id'].values
train_data = train_raw.drop(['id', 'is_iceberg'], axis=1)
test_raw = pd.read_json(path_test, precise_float=True)
test_data = test_raw.drop(['id'], axis=1)
test_ids = test_raw['id'].values

# convert input dataframes to images with 3 channels
def get_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        # make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        imgs.append(np.dstack((band_1, band_2, band_3)))

    return np.array(imgs)
 
X = get_imgs(train_data)
X_pred = get_imgs(test_data)
y = train_labels

# use the inclination angle as an auxiliary input for the CNN
# since it contains information for the iceberg/ship classification
# impute NaN's with median value and apply mean normalization
angle_train_temp = np.asarray(train_data['inc_angle'].replace('na', float('nan')))
angle_test_temp = np.asarray(test_data['inc_angle'].replace('na', float('nan')))

angle_train_med = np.nanmedian(angle_train_temp)
angle_test_med = np.nanmedian(angle_test_temp)

train_data['inc_angle'] = train_data['inc_angle'].replace('na', angle_train_med).astype(np.float32)
test_data['inc_angle'] = test_data['inc_angle'].replace('na', angle_test_med).astype(np.float32)

X_aux = np.asarray(train_data['inc_angle'])
X_aux_pred = np.asarray(test_data['inc_angle'])

X_aux = (X_aux - X_aux.mean()) / (X_aux.max() - X_aux.min())
X_aux_pred = (X_aux_pred - X_aux_pred.mean()) / (X_aux_pred.max() - X_aux_pred.min())

# define the architecture of the CNN
# convolutional layers
conv2d_1 = {'type':'conv2d','filters':64,'kernel_size':(3,3),'strides':(1,1)}
conv2d_2 = {'type':'conv2d','filters':128,'kernel_size':(3,3),'strides':(1,1)}
conv2d_3 = conv2d_2
conv2d_4 = conv2d_1

# pooling layers
max_pooling2d_1 = {'type':'max_pooling2d','pool_size':(3,3),'strides':(2,2)}
max_pooling2d_2 = {'type':'max_pooling2d','pool_size':(2,2),'strides':(2,2)}
max_pooling2d_3 = max_pooling2d_2
max_pooling2d_4 = max_pooling2d_3

# dropout layers
dropout_1 = {'type':'dropout','dropout':0.2}
dropout_2 = dropout_1
dropout_3 = {'type':'dropout','dropout':0.3}
dropout_4 = dropout_3

# fully connected layers
fc_1 = {'type':'fc','hidden_units':512,'batchnorm':False,'dropout':0.2}
fc_2 = {'type':'fc','hidden_units':128,'batchnorm':False,'dropout':0.2}

architecture = [conv2d_1,max_pooling2d_1,dropout_1,
                conv2d_2,max_pooling2d_2,dropout_2,
                conv2d_3,max_pooling2d_3,dropout_3,
                conv2d_4,max_pooling2d_4,dropout_4,
                fc_1,fc_2]

# create the CNN model object
# explanation:
# The model will use X as the training data with y as the target labels.
# The image data will be scaled using mean normalization (scale_data='meannorm').
# The training data will be augmented with rotation/shift/zoom and horizontal/
#   vertical flipping (augment_data='default').
# The angle data will be fed into the model just before the fully connected layers.
# The output of the last convolutional layers will be flattened (pooling='flat')
#   before the fully connected layers.
# Since we are dealing with a binary classification problem, we use the 'sigmoid'
#   output layer and 'binary_crossentropy' as a loss function.
# We are using early stopping on the validation data (20% of training data)
#   and training will occur on mini batches of size 32.
# More parameters can be set manually but are kept at their default values
#   in this example, see documentation in modules for more options.
                
cnn = CNN(X,y,architecture,X_aux=X_aux,X_pred=X_pred,X_aux_pred=X_aux_pred,train_id=train_ids,pred_id=test_ids,
          output_layer='sigmoid',loss_function='binary_crossentropy',pooling='flat',
          scale_data='meannorm',augment_data='default',early_stopping_epochs=30,
          custom_eval_stopping={'name':'binary_crossentropy','mode':'min','data':'val'},
          batch_size=32,val_size=0.2,save_dir=save_dir,model_name='statoil_cnn',target_col=['is_iceberg'])

# train the model, save the model, print the training history,
# make predictions and write predictions in csv file
# one epoch should take <10s on a GPU
cnn.train_model()
cnn.save_model()
cnn.print_training_history()
cnn.predict_model()
cnn.write_results()

# we can also run nfold cross-validation with n runs per fold, make predictions
# and out-of-fold predictions on the validation data, plot the training history,
# save the model and write the predictions in csv files all at once
cnn.nfolds = 5
cnn.runs = 3
cnn.write_output = True
cnn.cv_train_predict_model()
