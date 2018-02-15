
# For the transfer learning example we will use the kaggle dataset
# from the Statoil competition. In this competition, we need to classify
# ships and icebergs from satellite images.
# The data can be found through the link below:
# https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data

# directory where results will be saved
save_dir = '/home/arnaudvl/ML/kaggle/statoil/output/test/'

# specify train and test data paths
path_train = '/home/arnaudvl/ML/kaggle/statoil/input/train.json'
path_test = '/home/arnaudvl/ML/kaggle/statoil/input/test.json'

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

# pick model pre-trained on imagenet without the top layer
base_model = 'VGG16' # more info: https://arxiv.org/abs/1409.1556

# deeper layers of neural networks tend to learn more dataset specific features
# while the first layers learn more general features
# since we are trying to classify icebergs/ships from satellite images, our dataset
# is quite different from imagenet, so we remove deeper network layers from VGG16
# more info: https://arxiv.org/abs/1411.1792
remove_from_layer = 'block4_pool'

# add fully connected layers on top of base model
fc_1 = {'type':'fc','hidden_units':512,'batchnorm':False,'dropout':0.5}
fc_2 = {'type':'fc','hidden_units':256,'batchnorm':False,'dropout':0.5}
architecture = [fc_1,fc_2]

# create the CNN model object
# explanation:
# the model will use X as the training data with y as the target labels
# the training data will be augmented with rotation/shift/zoom and horizontal/
#   vertical flipping ('default' settings)
# global average pooling will be applied to the output of the last convolutional
#   layers (pooling='avg') before the fully connected layers
# since we are dealing with a binary classification problem, we use the 'sigmoid'
#   output layer and 'binary_crossentropy' as a loss function
# we are using early stopping on the validation data (20% of training data)
# training will occur on mini batches of size 32
# the only pre-processing VGG16 does is subtracting the mean RGB value, so we
#   won't apply normalization/standardization on the input images
# many more parameters can be set manually but are set to their default values
#   in this example, see documentation in modules for more options

cnntl = CNNTransferLearning(X,y,architecture,base_model,remove_from_layer=remove_from_layer,
                            X_pred=X_pred,train_id=train_ids,pred_id=test_ids,
                            output_layer='sigmoid',loss_function='binary_crossentropy',pooling='avg',
                            augment_data='default',early_stopping_epochs=30,
                            custom_eval_stopping={'name':'binary_crossentropy','mode':'min','data':'val'},
                            batch_size=32,val_size=0.2,save_dir=save_dir,model_name='statoil_vgg16',target_col=['is_iceberg'])
                            
# train the model and save the model weights
# one epoch should take <10s on a GPU
cnntl.train_model()
cnntl.save_model_weights()

# during the previous training phase, the weights of the VGG16 layers were frozen
# now we can fine tune layers with a lower learning rate
# note: if we do not freeze the layers in the first training phase, the
#       pre-trained VGG16 weights can be distorted because of the large 
#       gradients due to the randomly initialized top layer weights
# the finetuning below is just for illustrative purposes since the satellite 
# dataset is small and the model easily overfits
cnntl.trainable_layers = ['block4_conv3']
cnntl.learning_rate = 0.0001
cnntl.load_wgt = True
cnntl.train_model()

# we can also iteratively check which layers we should remove from the base model
remove_from_layer_tune = ['block5_conv2','block4_pool','block4_conv3']
cnntl = CNNTransferLearning(X,y,architecture,base_model,remove_from_layer_tune=remove_from_layer_tune,
                            X_pred=X_pred,train_id=train_ids,pred_id=test_ids,
                            output_layer='sigmoid',loss_function='binary_crossentropy',pooling='avg',
                            augment_data='default',early_stopping_epochs=30,
                            custom_eval_stopping={'name':'binary_crossentropy','mode':'min','data':'val'},
                            batch_size=32,val_size=0.2,save_dir=save_dir,model_name='statoil_vgg16_tune_layers',
                            target_col=['is_iceberg'])
cnntl.tune_layers_model()

# another option is to iteratively unfreeze and finetune layers from the base model
trainable_layers_tune = ['block5_conv3','block5_conv2','block5_conv1']
cnntl = CNNTransferLearning(X,y,architecture,base_model,trainable_layers_tune=trainable_layers_tune,
                            X_pred=X_pred,train_id=train_ids,pred_id=test_ids,
                            output_layer='sigmoid',loss_function='binary_crossentropy',pooling='max',
                            augment_data='default',early_stopping_epochs=10,
                            custom_eval_stopping={'name':'binary_crossentropy','mode':'min','data':'val'},
                            batch_size=32,val_size=0.2,save_dir=save_dir,model_name='statoil_vgg16_tune_layer_weights',
                            target_col=['is_iceberg'])
cnntl.tune_weights_model()
