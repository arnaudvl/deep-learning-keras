import pandas as pd

from dlkeras.dnn import DNN

# For the feedforward deep neural network example we will use the kaggle dataset
# from the Porto Seguro competition. In this competition, you’re challenged to 
# build a model that predicts the  probability that a driver will initiate an 
# auto insurance claim in the next year.
# The data can be found through the link below:
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

# directory where results will be saved
save_dir = '/.../output/'

# specify train and test data paths
path_train = '/.../input/train.csv'
path_test = '/.../input/test.csv'

# load data
train_raw = pd.read_csv(path_train,na_values=-1)
train_labels = train_raw['target'].values
train_ids = train_raw['id'].values
train_data = train_raw.drop(['id', 'target'], axis=1)
test_raw = pd.read_csv(path_test,na_values=-1)
test_data = test_raw.drop(['id'], axis=1)
test_ids = test_raw['id'].values

# remove uncorrelated features
unwanted = train_data.columns[train_data.columns.str.startswith('ps_calc_')]
train_data = train_data.drop(unwanted, axis=1)  
test_data = test_data.drop(unwanted, axis=1)

X = train_data
y = train_labels
X_pred = test_data

# specify architecture
fc_1 = {'hidden_units':80,'batchnorm':True,'dropout':0.3}
fc_2 = {'hidden_units':50,'batchnorm':True,'dropout':0.3}
fc_3 = {'hidden_units':20,'batchnorm':True,'dropout':0.2}
fc_4 = {'hidden_units':10,'batchnorm':True,'dropout':0.2}
architecture = [fc_1,fc_2,fc_3,fc_4]

# create neural net
# explanation:
# The model will use X as the training data with y as the target labels.
# Since we are dealing with a binary classification problem, we use the 'sigmoid'
#   output layer and 'binary_crossentropy' as a loss function.
# Missing values in the input data will be imputed by the median per column.
# The columns in cols_scaler will be scaled in the range [-1,1].
# One-hot-encoding will be applied to the categorical values (dummify='cat').
# We are using early stopping on the validation data (20% of training data),
#   using the roc-auc metric.
# Training will occur on mini batches of size 32.
# More parameters can be set manually but are set to their default values
#   in this example, see documentation in modules for more options.
cols_scaler = ['ps_ind_01','ps_ind_03','ps_ind_14','ps_ind_15',
               'ps_reg_01','ps_reg_02','ps_reg_03','ps_car_11',
               'ps_car_12','ps_car_13','ps_car_14','ps_car_15']
dnn = DNN(X,y,architecture,X_pred=X_pred,train_id=train_ids,pred_id=test_ids,
          output_layer='sigmoid',loss_function='binary_crossentropy',
          imputer='median',custom_eval_stopping={'name':'roc-auc','mode':'max','data':'val'},
          early_stopping_epochs=10,batch_size=32,val_size=0.2,dummify='cat',
          scaler={'type':'minmax','columns':cols_scaler},save_dir=save_dir,model_name='dnn')

# train and save the model, print the training history,
# make predictions and write predictions in csv file
dnn.train_model()
dnn.save_model()
dnn.print_training_history() # prints training/validation loss (binary_crossentropy),
                             # not necessarily the metric used for early stopping (roc-auc)
dnn.predict_model()
dnn.write_results()

# we can also run nfold cross-validation with n runs per fold, make predictions
# and out-of-fold predictions on the validation data, plot the training history,
# save the model and write the predictions in csv files all at once
dnn.nfolds = 5
dnn.runs = 3
dnn.write_output = True
dnn.cv_train_predict_model()
