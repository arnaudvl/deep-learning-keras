from keras.layers import concatenate, Dense, Dropout, Flatten, Activation, Conv2D, Input
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1
import numpy as np
from sklearn.model_selection import StratifiedKFold

from dlkeras.nn import NN

class CNN(NN):
    
    """
    Create and train models, and make predictions for convolutional neural nets using keras with tensorflow backend.
    
    Main functions:
        - init_model: set up and compile convolutional neural net
        - train_model
        - predict_model
        - cv_train_predict_model: run nfold CV on neural net and make predictions on test data
    """
    
    def __init__(self,X,y,architecture,X_aux=None,output_layer='softmax',optimizer_type='adam',
                 loss_function='categorical_crossentropy',metrics=['accuracy'],learning_rate=0.001,
                 learning_rate_decay=0.,learning_rate_factor=0.1,num_epochs=10000,early_stopping_epochs=100,
                 custom_eval_stopping={'name':'roc-auc','mode':'max','data':'val'},nfolds=5,
                 batch_size=64,pooling='max',runs=3,val_size=0.2,random_state=1,X_pred=None,X_aux_pred=None,
                 train_id=None,pred_id=None,scale_data=None,augment_data={},scale_pre_concat=False,
                 load_wgt=False,print_out=True,print_training=True,write_output=False,save_dir=None,
                 model_name=None,target_col=['target'],id_col='id'):
        
        """
        Arguments:
            num_epochs -- int, number of passes over training set, default=10000
            nfolds -- int, number of folds for CV, default=5
            batch_size -- int, number of training samples in each step, default=64
            pooling -- str, pooling method before fully connected layers, options=None,'max','avg' or 'flat', default='avg'
            random_state -- int, initiate random state to generate CV folds, default=1
            X_pred -- np array, data on which predictions are made, default=None
            X_aux_pred -- np array, auxiliary data on which predictions are made, default=None
            runs -- int, number of runs over each fold in CV with different random seed, default=3
            train_id -- np array, id's of training data, default=None
            scale_data -- str, data scaling with scaler 'minmax' (feature range=(0,1)),'std','meannorm' or None, default=None
            augment_data -- dict, contains arguments for ImageDataGenerator, can also take 'default' as input,
                            https://keras.io/preprocessing/image/, default={}
            scale_pre_concat -- bool, can apply batchnorm to model output after pooling layer in case of multiple inputs, default=False
            load_wgt -- bool, can load model weights from file "model_name+'_weights.h5'" when initiating the model, default=False
            print_out -- bool, print out intermediate results of CV, default=True
            print_training -- bool, print graph with training progress during CV over loss function, default=True
            
            See class NN for documentation of other arguments. 
        """
        
        NN.__init__(self,X,y,architecture,X_aux=X_aux,output_layer=output_layer,optimizer_type=optimizer_type,
                    loss_function=loss_function,metrics=metrics,learning_rate=learning_rate,
                    learning_rate_decay=learning_rate_decay,learning_rate_factor=learning_rate_factor,
                    early_stopping_epochs=early_stopping_epochs,custom_eval_stopping=custom_eval_stopping,
                    val_size=val_size,pred_id=pred_id,write_output=write_output,save_dir=save_dir,
                    model_name=model_name,target_col=target_col,id_col=id_col)
        
        self.num_epochs = num_epochs
        self.nfolds = nfolds
        self.batch_size = batch_size
        self.runs = runs
        self.pooling = pooling
        self.random_state = random_state
        self.X_pred = X_pred
        self.X_aux_pred = X_aux_pred
        self.train_id = train_id
        self.scale_data = scale_data
        self.augment_data = augment_data
        self.scale_pre_concat = scale_pre_concat
        self.load_wgt = load_wgt
        self.print_out = print_out
        self.print_training = print_training
        
        # default values for data augmentation if used
        self._augment_data_default = {'featurewise_center':False,
                                      'samplewise_center':False,
                                      'featurewise_std_normalization':False,
                                      'samplewise_std_normalization':False,
                                      'zca_whitening':False,
                                      'zca_epsilon':1e-6,
                                      'rotation_range':20,
                                      'width_shift_range':0.2,
                                      'height_shift_range':0.2,
                                      'shear_range':0.,
                                      'zoom_range':0.2,
                                      'channel_shift_range':0.,
                                      'cval':0.,
                                      'horizontal_flip':True,
                                      'vertical_flip':True,
                                      'rescale':None,
                                      'preprocessing_function':None}
        
        self._data_generator = None
        self._data_generator_pred = None
        self._is_scaled = False
        self._params_fit = None
        self._cv = False
        self._print_summary = False
    
    
    def scale_std(self,X):
        """
        Standardize X per channel.
        """
        X_scale_std = X.copy()
        i = 0
        for item in X_scale_std:
            for j in range(item.shape[2]):
                layer = X_scale_std[i][:,:,j]
                X_scale_std[i][:,:,j] = (layer - layer.mean()) / layer.std()
            i+=1
        return X_scale_std
    
    
    def scale_minmax(self,X,feature_range=[0,1]):
        """
        Apply minmax scaling per channel in feature range.
        """
        X_scale_minmax = X.copy()
        i = 0
        for item in X_scale_minmax:
            for j in range(item.shape[2]):
                layer = X_scale_minmax[i][:,:,j]
                X_scale_minmax[i][:,:,j] = ((layer - layer.min()) / (layer.max() - layer.min()) *
                                            (feature_range[1] - feature_range[0]) + feature_range[0])
            i+=1
        return X_scale_minmax
    
    def scale_meannorm(self,X):
        """
        Apply mean normalization per channel.
        """
        X_scale_meannorm = X.copy()
        i = 0
        for item in X_scale_meannorm:
            for j in range(item.shape[2]):
                layer = X_scale_meannorm[i][:,:,j]
                X_scale_meannorm[i][:,:,j] = (layer - layer.mean()) / (layer.max() - layer.min())
            i+=1
        return X_scale_meannorm
    
    
    def _apply_scaling(self):
        """
        Scale input data:
            - 'std': standardization
            - 'minmax': minmax scaling in [0,1] area
            - 'meannorm': mean normalization
        
        Beware: scaling is only applied on input but not again on afterwards augmented data.
        """
        if self.scale_data=='std':
            
            self.X = self.scale_std(self.X)
            if self.X_pred is not None:
                self.X_pred = self.scale_std(self.X_pred)
                
        elif self.scale_data=='minmax': #'minmax' is only applied on input but not augmented data
        
            self.X = self.scale_minmax(self.X,feature_range=[0,1])
            if self.X_pred is not None:
                self.X_pred = self.scale_minmax(self.X_pred,feature_range=[0,1])
        
        elif self.scale_data=='meannorm': #'meannorm' is only applied on input but not augmented data
            
            self.X = self.scale_meannorm(self.X)
            if self.X_pred is not None:
                self.X_pred = self.scale_meannorm(self.X_pred)
        
        self._is_scaled = True
        return self
  
    
    def _create_data_generator(self):
        """
        Create data generator and fit on training data if needed.
        """
        if not self._data_generator:
            if not self.augment_data:
                return self
            elif self.augment_data=='default':
                self.augment_data = self._augment_data_default
            
            self._data_generator = ImageDataGenerator(**self.augment_data)
            
            fit_list = ['featurewise_center','featurewise_std_normalization',
                        'samplewise_center','samplewise_std_normalization',
                        'zca_whitening']
            for item in fit_list:
                if self.augment_data[item]:
                    self._data_generator.fit(self.X)
                    if self.X_pred is not None:
                        augment_data_pred = {}
                        for itm in fit_list:
                            if self.augment_data[itm]:
                                augment_data_pred[itm] = True
                        self._data_generator_pred = ImageDataGenerator(**augment_data_pred)
                        self._data_generator_pred.fit(self.X_pred)
                    break
                
        return self
    
    
    def _apply_data_generator(self):
        """
        Run data generator, taking fixed auxiliary input data into account.
        """                          
        if self.X_train is None:
            x = self.X
            y = self.y
        else:
            x = self.X_train
            y = self.y_train
                
        if self.X_aux is None:
            gen = self._data_generator.flow(x,y,batch_size=self.batch_size)
            while True:
                xi = gen.next()
                yield xi[0],xi[1]
        else:
            if self.X_aux_train is None:
                x_aux = self.X_aux
            else:
                x_aux = self.X_aux_train
            
            seed = np.random.randint(100,size=1)[0]
            gen1 = self._data_generator.flow(x,y,batch_size=self.batch_size,seed=seed)
            gen2 = self._data_generator.flow(x,x_aux,batch_size=self.batch_size,seed=seed)
            
            while True:
                x1i = gen1.next()
                x2i = gen2.next()
                # assert arrays are equal
                np.testing.assert_array_equal(x1i[0],x2i[0])
                yield [x1i[0],x2i[1]], x1i[1]
    
    
    def init_model(self):
        """
        Set up and compile convolutional neural net architecture.
        """
        self._get_default_values_architecture()
        
        if self.X_train is not None:
            input_shape = self.X_train.shape[1:]
        else:
            input_shape = self.X.shape[1:]
        
        main_input = Input(shape=input_shape)
        
        ilayer = 0
        add_aux = False
        while ilayer<len(self.architecture): # define CNN architecture
            
            if ilayer==0:
                if self.architecture[ilayer]['type']=='conv2d':
                    x = Conv2D(self.architecture[ilayer]['filters'],
                               kernel_size=self.architecture[ilayer]['kernel_size'],
                               strides=self.architecture[ilayer]['strides'],
                               padding=self.architecture[ilayer]['padding'],
                               activation=self.architecture[ilayer]['activation'],
                               kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                               bias_initializer=self.architecture[ilayer]['bias_initializer'],
                               kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1']))(main_input)
                    ilayer+=1
                    continue
                else:
                    raise ValueError('First layer needs to be a "conv2d" layer.')
            
            # 2d convolution layer
            if self.architecture[ilayer]['type']=='conv2d':
                x = Conv2D(self.architecture[ilayer]['filters'],
                           kernel_size=self.architecture[ilayer]['kernel_size'],
                           strides=self.architecture[ilayer]['strides'],
                           padding=self.architecture[ilayer]['padding'],
                           activation=self.architecture[ilayer]['activation'],
                           kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                           bias_initializer=self.architecture[ilayer]['bias_initializer'],
                           kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1']))(x)
            
            # max pooling layer
            elif self.architecture[ilayer]['type']=='max_pooling2d':
                x = MaxPooling2D(pool_size=self.architecture[ilayer]['pool_size'],
                                 strides=self.architecture[ilayer]['strides'],
                                 padding=self.architecture[ilayer]['padding'])(x)
            
            # average pooling layer
            elif self.architecture[ilayer]['type']=='avg_pooling2d':
                x = AveragePooling2D(pool_size=self.architecture[ilayer]['pool_size'],
                                     strides=self.architecture[ilayer]['strides'],
                                     padding=self.architecture[ilayer]['padding'])(x)
            
            # global max pooling layer
            elif self.architecture[ilayer]['type']=='global_max_pooling2d':
                x = GlobalMaxPooling2D()(x)
            
            # global average pooling layer
            elif self.architecture[ilayer]['type']=='global_avg_pooling2d':
                x = GlobalAveragePooling2D()(x)
            
            # dropout
            elif self.architecture[ilayer]['type']=='dropout':
                x = Dropout(self.architecture[ilayer]['dropout'])(x)

            # fully connected layer
            elif self.architecture[ilayer]['type']=='fc':
                
                # flatten data if not done yet using global max or avg pooling
                if len(x.shape)>2:
                    # pooling/flattening method
                    if self.pooling=='avg':
                        x = GlobalAveragePooling2D()(x)
                    elif self.pooling=='max':
                        x = GlobalMaxPooling2D()(x)
                    elif self.pooling=='flat':
                        x = Flatten()(x)
                    else:
                        raise ValueError('Only "max", "avg" and "flat" are valid flattening/pooling methods.')
                
                # add in auxiliary input data
                if self.X_aux is not None and not add_aux:
                    # add batchnorm option in case auxiliary input is scaled
                    if self.scale_pre_concat:
                        x = BatchNormalization()(x)
                    
                    # merge model with auxiliary input
                    if self.X_aux.shape[1:]==():
                        aux_shape = (1,)
                    else:
                        aux_shape = self.X_aux.shape[1:]
                    
                    aux_input = Input(shape=aux_shape, name='aux_input')
                    x = concatenate([x,aux_input])
                    add_aux = True
                
                x = Dense(self.architecture[ilayer]['hidden_units'],
                          kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                          bias_initializer=self.architecture[ilayer]['bias_initializer'],
                          kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1']))(x)
                if self.architecture[ilayer]['batchnorm']:
                    x = BatchNormalization()(x)
                x = Activation(self.architecture[ilayer]['activation'])(x)
                x = Dropout(self.architecture[ilayer]['dropout'])(x)
            
            else:
                raise ValueError('Valid types for CNN architecture: "conv2d","max_pooling2d","avg_pooling2d","fc",' \
                                 '"global_max_pooling2d","global_avg_pooling2d","dropout".')
            
            ilayer+=1
        
        n_classes = self._num_classes()
        if self.output_layer=='sigmoid':
            if n_classes==2:
                predictions = Dense(1,activation=self.output_layer)(x) # output layer
            else:
                raise ValueError('sigmoid output layer suitable for binary classification, but %i classes detected.' \
                                    %(n_classes))
        else:
            predictions = Dense(n_classes,activation=self.output_layer)(x) # output layer

        # add auxiliary input
        if self.X_aux is not None:
            model_input = [main_input,aux_input]
        else:
            model_input = main_input
        
        # create model
        model_cnn = Model(inputs=model_input,outputs=predictions)
        
        # load weights if specified
        if self.load_wgt:
            model_cnn.load_weights(self.model_name + '_weights.h5')
        
        # compile model
        model_cnn.compile(optimizer=self._get_optimizer(),metrics=self.metrics,loss=self.loss_function)
        
        if not self._print_summary:
            model_cnn.summary() # display model architecture
            self._print_summary = True
        
        return model_cnn
    
    
    def _get_params_fit(self,i=0,run=0):
        """
        Set parameters to train the model.
        """   
        self._params_fit = {'epochs':self.num_epochs,
                            'batch_size':self.batch_size,
                            'verbose':self._set_verbose(),
                            'shuffle':True,
                            'callbacks':self._callbacks(i=i,run=run)}
                            
        if self.X_val is not None:
            if self.X_aux_val is None:
                self._params_fit['validation_data'] = (self.X_val,self.y_val)
            else:
                self._params_fit['validation_data'] = ([self.X_val,self.X_aux_val],self.y_val)
                
        if bool(self.augment_data):
            
            if self.X_train is not None:
                steps_per_epoch = len(self.X_train) // self.batch_size
            else:
                steps_per_epoch = len(self.X) // self.batch_size
            
            self._params_fit['steps_per_epoch'] = steps_per_epoch
            
            del self._params_fit['batch_size']
            del self._params_fit['shuffle']
            
        return self
    
    
    def train_model(self,i=0,run=0):
        """
        Train model.
        """
        self.nn_model = self.init_model() # initiate model
        
        if self.scale_data is not None and not self._is_scaled:
            self._apply_scaling()
        
        if bool(self.augment_data) and not self._data_generator:
            self._create_data_generator()
        
        if self.val_size>0 and self.X_val is None: # need to create validation data
            self._train_test_split()
        
        self._get_params_fit(i=i,run=run)
        
        if not self.augment_data: # train without augmenting data
            
            params = self._params_fit
                
            if self.X_train is not None and self.X_val is not None:
                # already have training and validation data
                if self.X_aux_train is not None:
                    params['x'] = [self.X_train,self.X_aux_train]
                else:
                    params['x'] = self.X_train
                params['y'] = self.y_train
            else: # don't use validation data
                if self.X_aux is not None:
                    params['x'] = [self.X,self.X_aux]
                else:
                    params['x'] = self.X
                params['y'] = self.y
            
            self.nn_fit = self.nn_model.fit(**params)
                
        elif bool(self.augment_data):
             
            self.nn_fit = self.nn_model.fit_generator(self._apply_data_generator(),**self._params_fit)
            
        return self
    
    
    def predict_model(self,model_type='last',i=0,run=0):
        """
        Make predictions on X_pred.
        """
        if model_type=='last': # predict on best model weights
            del self.nn_model
            self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        
        if self.X_pred is not None:
            if self._data_generator_pred is None:
                if self.X_aux_pred is None:
                    self.y_pred = self.nn_model.predict(self.X_pred,verbose=0)
                else:
                    self.y_pred = self.nn_model.predict([self.X_pred,self.X_aux_pred],verbose=0)
            else:
                if self.X_aux_pred is None:
                    self.y_pred = self.nn_model.predict_generator(self._data_generator_pred.flow(self.X_pred,
                                                                  batch_size=len(self.X_pred),shuffle=False),steps=1)
                else:
                    self.y_pred = self.nn_model.predict_generator(self._data_generator_pred.flow([self.X_pred,self.X_aux_pred],
                                                                  batch_size=len(self.X_pred),shuffle=False),steps=1)
        else:
            print('No values "X_pred" to perform prediction on.')
        return self

    
    def cv_train_predict_model(self):
        """
        Run nfold CV on the neural net.
        """
        self._cv = True
        skf = StratifiedKFold(n_splits=self.nfolds,shuffle=True,random_state=self.random_state)
        start_time = self._timer(None)
        
        if self.scale_data is not None and not self._is_scaled:
            self._apply_scaling()
        
        if bool(self.augment_data) and not self._data_generator:
            self._create_data_generator()
        
        for i, (train_index, test_index) in enumerate(skf.split(self.X,self.y)):
            self.X_train, self.X_val = self.X[train_index], self.X[test_index]
            self.y_train, self.y_val = self.y[train_index], self.y[test_index]
            if self.X_aux is not None:
                self.X_aux_train, self.X_aux_val = self.X_aux[train_index], self.X_aux[test_index]
            train_ids, val_ids = self.train_id[train_index], self.train_id[test_index]
            
            self.init_model # set up model initialization function
            
            # repeat runs for each fold with different seed
            for run in range(self.runs):
                print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
                np.random.seed()
                
                self.train_model(i=i,run=run) # train neural net
                
                if self.print_training: # graph with progress of model performance each run
                    self.print_training_history(i=i,run=run)
                
                # want best saved model, not last where training stopped
                # delete last model instance, load last saved checkpoint
                del self.nn_model
                self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
                if self.X_aux_train is not None:
                    scores_val_run = self.nn_model.predict([self.X_val,self.X_aux_val],verbose=0)
                    y_pred_run = self.nn_model.predict([self.X_pred,self.X_aux_pred],verbose=0)
                else:
                    scores_val_run = self.nn_model.predict(self.X_val,verbose=0)
                    y_pred_run = self.nn_model.predict(self.X_pred,verbose=0)
                scores_val_run = np.reshape(scores_val_run,self.y_val.shape)
                
                if self.print_out:
                    print(' Fold %d Run %d %s: %.5f' % ((i + 1), (run + 1), self._val_loss,self._calc_metric(self.y_val,scores_val_run)))
                
                if run==0:
                    scores_val = scores_val_run
                    y_pred = y_pred_run
                else:
                    scores_val += scores_val_run
                    y_pred += y_pred_run
            
            scores_val = scores_val / self.runs
            y_pred = y_pred / self.runs
            
            if self.print_out:
                print(' Fold %d %s: %.5f' % ((i + 1), self._val_loss,self._calc_metric(self.y_val,scores_val)))
                self._timer(start_time=start_time)
            
            if i==0:
                pred = y_pred
                self._y_val_folds = self.y_val
                self._scores_val_folds = scores_val
                self._val_ids_folds = val_ids
                score = self._calc_metric(self.y_val,scores_val)
            else:
                pred += y_pred
                self._y_val_folds = np.concatenate((self._y_val_folds, self.y_val), axis=0)
                self._scores_val_folds = np.concatenate((self._scores_val_folds, scores_val), axis=0)
                self._val_ids_folds = np.concatenate((self._val_ids_folds, val_ids), axis=0)
                score += self._calc_metric(self.y_val,scores_val)
            
        self.y_pred = pred / self.nfolds
        
        self._score_oof = self._calc_metric(self._y_val_folds,self._scores_val_folds)
        
        if self.print_out:
            print('\n Average %s: %.5f' % (self._val_loss, score/self.nfolds))
            print(' Out-of-fold %s: %.5f' % (self._val_loss,self._score_oof))
            
        if self.write_output:
            self.write_results() # write oof and test predictions in csv files
            self.save_model(model_type='best') # only saves best model from last run
        
        self._cv = False
        return self
