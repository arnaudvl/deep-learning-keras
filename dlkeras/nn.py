from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.losses import mean_squared_error,mean_absolute_error,categorical_crossentropy,binary_crossentropy
from keras.models import load_model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class NN():
    
    """
    Class with reusable functions for Feedforward NN and CNN.
    """
    
    def __init__(self,X,y,architecture,X_aux=None,output_layer='softmax',optimizer_type='adam',
                 loss_function='categorical_crossentropy',metrics=['accuracy'],learning_rate=0.001,
                 learning_rate_decay=0.,learning_rate_factor=0.1,early_stopping_epochs=100,
                 custom_eval_stopping={'name':'roc-auc','mode':'max','data':'val'},
                 val_size=0.2,pred_id=None,write_output=False,save_dir=None,model_name=None,
                 target_col=['target'],id_col='id'):
        
        """
        Arguments:
            X -- np array or pd dataframe for DNN, training data
            y -- np array, training labels
            architecture -- list, list of layers in NN architecture. Each layer is specified with a dict.
                                  Example: [conv2d_1,max_pooling2d_1,dropout_1,conv2d_2,max_pooling2d_2,dropout_2,fc_1,fc_2]
                                  where
                                  conv2d_1 = {'type':'conv2d','filters':64,'kernel_size':(3,3),'strides':(1,1),
                                              'padding':'valid','activation':'relu','kernel_initializer':'glorot_uniform',
                                              'bias_initializer':'zeros','kernel_regularizer_l1':0.2}
                                  max_pooling2d_1 = {'type':'max_pooling2d','pool_size':(2,2),'strides':2,'padding':'valid'}
                                  dropout_1 = {'type':'dropout','dropout':0.3}
                                  fc_1 = {'type':'fc','hidden_units':512,'activation':'relu',
                                          'kernel_initializer':'glorot_uniform',
                                          'bias_initializer':'zeros','kernel_regularizer_l1':1.,'dropout':0.2}
            X_aux -- np array, auxiliary input for multi-input models, default=None
            output_layer -- str, output layer type (https://keras.io/activations), default='softmax'
            optimizer_type -- str, optimizer, options='adam','sgd','rmsprop','adagrad','adadelta',
                                   'adamax','nadam', default='adam'
            loss_function -- str, loss function to evaluate (https://keras.io/losses/),
                                  default='categorical_crossentropy'
            metrics -- list, performance metric(s) to evaluate, default=['accuracy']
            learning_rate -- float, default=0.001
            learning_rate_decay -- float, decay rate of learning rate in optimizer, default=0.
            learning_rate_factor -- float, learning rate multiplied by factor if no improvement over evaluation function
                                           over 50% of early stopping rounds, default=0.1
            early_stopping_epochs -- int, number of epochs after which early stopping occurs if
                                          stopping metric does not improve, default=100
            custom_eval_stopping -- dict, custom metric for early stopping, with keys "name" (name of metric),
                                          "mode" (max or min the metric), "data" (train or val data used),
                                          default={'name':'roc-auc','mode':'max','data':'val'}
            val_size -- float, fraction of training data used for validation, default=0.2
            pred_id -- np array, id's of prediction data, default=None
            write_output -- bool, write CV results in csv files and store model, default=False
            save_dir -- str, directory for model to be saved in, default=None
            model_name -- str, model name, default=None
            target_col -- list, name of target column(s), default=['target']
            id_col -- str, name of id column, default='id'
        """
        
        self.X = X
        self.y = y
        self.architecture = architecture
        self.X_aux = X_aux
        self.output_layer = output_layer
        self.optimizer_type = optimizer_type
        self.loss_function = loss_function
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_factor = learning_rate_factor
        self.early_stopping_epochs = early_stopping_epochs
        self.custom_eval_stopping = custom_eval_stopping
        self.val_size = val_size
        self.pred_id = pred_id
        self.write_output = write_output
        self.save_dir = save_dir
        self.model_name = model_name
        self.target_col = target_col
        self.id_col = id_col
        
        if type(self.custom_eval_stopping)==dict:
            self._val_loss = self.custom_eval_stopping['name']
            if self.custom_eval_stopping['name']==self.loss_function:
                self._val_loss_callback = 'val_loss'
            else:
                self._val_loss_callback = (self.custom_eval_stopping['name'] + '-' + 
                                           self.custom_eval_stopping['data'])
            self._mode = self.custom_eval_stopping['mode']
        else:
            self._val_loss = 'roc-auc'
            self._val_loss_callback = self.metrics[0] + '-val'
            self._mode = 'max'
        
        # default values for conv2d, max_pooling2d and fc layers
        self._conv2d_default = {'strides':(1,1),
                                'padding':'valid',
                                'activation':'relu',
                                'kernel_initializer':'glorot_uniform',
                                'bias_initializer':'zeros',
                                'kernel_regularizer_l1':0.}
        self._pooling2d_default = {'padding':'valid'}
        self._dropout_default = {'dropout':0.}
        self._fc_default = {'activation':'relu',
                            'kernel_initializer':'glorot_uniform',
                            'bias_initializer':'zeros',
                            'kernel_regularizer_l1':0.,
                            'kernel_regularizer_l2':0.,
                            'dropout':0.,
                            'batchnorm':True}
        
        # learning rate reduction with learning_rate_factor if no improvement 
        # after fraction of early stopping steps
        self._early_stopping_lr = 0.5
        
        self.nn_model = None
        self.nn_fit = None
        self.y_pred = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_aux_train = None
        self.X_aux_val = None
        self._y_val_folds = None
        self._scores_val_folds = None
        self._val_ids_folds = None
        self._score_oof = None
        
    
    def _timer(self,start_time=None):
        """
        Return time spent over calculations.
        """
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' %
                  (thour, tmin, round(tsec, 2)))
    
    
    def _train_test_split(self):
        """
        If no CV is done but part of the data is set apart for validation,
        apply train_test_split.
        """
        if self.val_size>0.:
            if self.X_aux is not None:
                self.X_train, self.X_val, self.y_train, self.y_val, self.X_aux_train, self.X_aux_val = \
                    train_test_split(self.X, self.y, self.X_aux, test_size=self.val_size, shuffle=True)
            else:
                self.X_train, self.X_val, self.y_train, self.y_val = \
                    train_test_split(self.X, self.y, test_size=self.val_size, shuffle=True)
        return self
    
    
    def _calc_metric(self,y_true,y_pred):
        """
        Calculate evaluation metric.
        
        Supports: "roc-auc","norm-gini","mean_squared_error","mean_absolute_error",
                  "categorical_crossentropy","binary_crossentropy".
        """
        if self._val_loss=='roc-auc':
            metric = roc_auc_score(y_true, y_pred)
        elif self._val_loss=='norm-gini':
            metric = (2 * roc_auc_score(y_true, y_pred)) - 1
        elif self._val_loss=='mean_squared_error':
            metric = K.eval(mean_squared_error(K.variable(y_true), K.variable(y_pred)))
        elif self._val_loss=='mean_absolute_error':
            metric = K.eval(mean_absolute_error(K.variable(y_true), K.variable(y_pred)))
        elif self._val_loss=='categorical_crossentropy':
            metric = K.eval(categorical_crossentropy(K.variable(y_true), K.variable(y_pred)))
        elif self._val_loss=='binary_crossentropy':
            metric = K.eval(binary_crossentropy(K.variable(y_true), K.variable(y_pred)))
        else:
            raise ValueError('Invalid value for "custom_eval_stopping["name"], "roc-auc","norm-gini","mean_squared_error", \
                             "mean_absolute_error","categorical_crossentropy","binary_crossentropy" supported.')
        return metric
    
    
    def _set_verbose(self):
        """
        Set print option.
        """
        if self.print_out:
            verbose = 2
        else:
            verbose = 0
        return verbose
    
    
    def _get_optimizer(self):
        """
        Define optimizer for the neural net.
        """
        if self.optimizer_type=='adam':
            optimizer = Adam(lr=self.learning_rate,decay=self.learning_rate_decay)
        elif self.optimizer_type=='sgd':
            optimizer = SGD(lr=self.learning_rate, momentum=0.0, decay=self.learning_rate_decay, nesterov=False)
        elif self.optimizer_type=='rmsprop':
            optimizer = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-08, decay=self.learning_rate_decay)
        elif self.optimizer_type=='adagrad':
            optimizer = Adagrad(lr=self.learning_rate, epsilon=1e-08, decay=self.learning_rate_decay)
        elif self.optimizer_type=='adadelta':
            optimizer = Adadelta(lr=self.learning_rate, rho=0.95, epsilon=1e-08, decay=self.learning_rate_decay)
        elif self.optimizer_type=='adamax':
            optimizer = Adamax(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.learning_rate_decay)
        elif self.optimizer_type=='nadam':
            optimizer = Nadam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        else:
            raise ValueError('"optimizer_type" %s is not valid and needs to be one of: "adam","sgd", \
                             "rmsprop","adagrad","adadelta","adamax","nadam".' %(self.optimizer_type))
        return optimizer
    
    
    def _set_default(self,_a,_b):
        """
        helper function to get default values if input argument not specified
        """
        for key,value in _b.items():
            try:
                _a[key]
            except KeyError:
                _a[key] = value
        return _a
    
    
    def _get_default_values_architecture(self):
        """
        Get default values for layers in CNN if not specified in input architecture.
        """
        ilayer = 0
        while ilayer<len(self.architecture):
            if self.architecture[ilayer]['type']=='conv2d':
                self.architecture[ilayer] = self._set_default(self.architecture[ilayer],self._conv2d_default)
            elif (self.architecture[ilayer]['type']=='max_pooling2d' or 
                    self.architecture[ilayer]['type']=='avg_pooling2d'):
                self.architecture[ilayer] = self._set_default(self.architecture[ilayer],self._pooling2d_default)
            elif self.architecture[ilayer]['type']=='fc':
                self.architecture[ilayer] = self._set_default(self.architecture[ilayer],self._fc_default)
            elif self.architecture[ilayer]['type']=='dropout':
                self.architecture[ilayer] = self._set_default(self.architecture[ilayer],self._dropout_default)
            ilayer+=1
        return self
    
    
    def _num_classes(self):
        """
        Get number of target classes.
        """
        unique, counts = np.unique(self.y, return_counts=True)
        return len(unique)
    
    
    def _callbacks(self,i=0,run=0):
        """
        Define callbacks to log performance metrics, apply early stopping,
        reduce the learning rate and save best models during training.
        """
        os.chdir(self.save_dir)
        
        callbacks = [EarlyStopping(monitor=self._val_loss_callback,
                                   patience=self.early_stopping_epochs,
                                   verbose=1,
                                   mode=self._mode),
                     ModelCheckpoint(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                                     monitor=self._val_loss_callback,
                                     verbose=1,
                                     mode=self._mode,
                                     save_best_only=True),
                    ReduceLROnPlateau(monitor=self._val_loss_callback,
                                      patience=int(self.early_stopping_epochs*self._early_stopping_lr),
                                      factor=self.learning_rate_factor,
                                      verbose=1,
                                      mode=self._mode,
                                      epsilon=0.0001)]
        
        if self.X_train is not None and self.X_val is not None:
        
            if self.X_aux_train is not None and self.X_aux_val is not None:
                training_data = (self.X_train,self.y_train,self.X_aux_train)
                validation_data = (self.X_val,self.y_val,self.X_aux_val)
            else:
                training_data = (self.X_train,self.y_train)
                validation_data = (self.X_val,self.y_val)
            
            callbacks.insert(0,MetricsLog(training_data=training_data,
                                          validation_data=validation_data,
                                          metric=self._val_loss))
        else:
            
            if self.X_aux is not None:
                training_data=(self.X,self.y,self.X_aux)
            else:
                training_data = (self.X,self.y)
            
            callbacks.insert(0,MetricsLog(training_data=training_data,
                                          metric=self._val_loss))
        
        return callbacks
    
    
    def print_training_history(self,i=0,run=0):
        """
        Print graph of training vs. validation performance of loss function.
        """
        fig = plt.figure()
        plt.plot(self.nn_fit.history['loss'])
        plt.plot(self.nn_fit.history['val_loss'])
        plt.title('model loss vs epoch fold ' + str('%02d' % (i + 1)) + ' run ' + str('%02d' % (run + 1)))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        if self.write_output and self.save_dir is not None and self.model_name is not None:
            os.chdir(self.save_dir)
            fig.savefig(self.model_name + '-performance-fold-' + str('%02d' % (i + 1)) + \
                        '-run-' + str('%02d' % (run + 1)) + '.png')
        return self
    
    
    def save_model(self,model_type='last',i=0,run=0):
        """
        Save trained model into a HDF5 file, containing:
        - model architecture
        - model weights
        - training config (loss,optimizer)
        - state optimizer where you left off
        """
        if model_type=='last': # only load best model weights
            del self.nn_model
            self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        
        if self.save_dir is not None and self.model_name is not None:
            os.chdir(self.save_dir)
            self.nn_model.save(self.model_name + '.h5')
        else:
            print('Need save_dir and model_name to save model.')
        return
    
    
    def save_model_weights(self,model_type='last',i=0,run=0):
        """
        Save trained model into a HDF5 file, containing:
        - model weights
        """
        if model_type=='last': # only load best model weights
            del self.nn_model
            self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        
        if self.save_dir is not None and self.model_name is not None:
            os.chdir(self.save_dir)
            self.nn_model.save_weights(self.model_name + '_weights.h5')
        else:
            print('Need save_dir and model_name to save model weights.')
        return
    
    
    def load_model(self):
        """
        Load trained model.
        """
        if self.save_dir is not None and self.model_name is not None:
            os.chdir(self.save_dir)
            self.nn_model = load_model(self.model_name + '.h5')
        else:
            print('Need save_dir and model_name to load model.')
        return
    
    
    def write_results(self):
        """
        Write the CV out-of-fold and predictions in csv files.
        """
        print('\n Writing results')
        now = datetime.now()
        
        # oof
        if self._scores_val_folds is not None:
            oof_result = pd.DataFrame(self._y_val_folds, columns=self.target_col)
            oof_result[self.id_col] = self._val_ids_folds
            oof_result.sort_values(self.id_col, ascending=True, inplace=True)
            oof_result = oof_result.set_index(self.id_col)
            sub_file = self.model_name + '-oof-' + str(self._score_oof) + '-' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
            print('\n Writing out-of-fold file:  %s' % sub_file)
            oof_result.to_csv(sub_file, index=True, index_label=self.id_col)
        
        # predictions
        if self.y_pred is not None:
            result = pd.DataFrame(self.y_pred, columns=self.target_col)
            result[self.id_col] = self.pred_id
            result = result.set_index(self.id_col)
            print('\n First 10 lines of average prediction over the folds:\n')
            print(result.head(10))
            if self._score_oof is not None:
                sub_file = self.model_name + '-prediction-' + str(self._score_oof) + '-' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
            else:
                sub_file = self.model_name + '-prediction-' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
            print('\n Writing prediction:  %s' % sub_file)
            result.to_csv(sub_file, index=True, index_label=self.id_col)
        
        return


class MetricsLog(Callback):
    
    def __init__(self,training_data,validation_data=(None,None),metric='roc-auc'):
        
        self.X_train = training_data[0]
        self.y_train = training_data[1]
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        
        if len(training_data)==3:
            self.X_aux_train = training_data[2]
        else:
            self.X_aux_train = None
            
        if len(validation_data)==3:
            self.X_aux_val = validation_data[2]
        else:
            self.X_aux_val = None
        
        self._metric = metric
    
    
    def _get_metric(self,y_true,y_pred):
        """
        Calculate metric being logged.
        
        Supports: "roc-auc","norm-gini","mean_squared_error","mean_absolute_error",
                  "categorical_crossentropy","binary_crossentropy".
        """
        if self._metric=='roc-auc':
            metric = roc_auc_score(y_true, y_pred)
        elif self._metric=='norm-gini':
            metric = (2 * roc_auc_score(y_true, y_pred)) - 1
        elif self._metric=='mean_squared_error':
            metric = K.eval(mean_squared_error(K.variable(y_true), K.variable(y_pred)))
        elif self._metric=='mean_absolute_error':
            metric = K.eval(mean_absolute_error(K.variable(y_true), K.variable(y_pred)))
        elif self._metric=='categorical_crossentropy':
            metric = K.eval(categorical_crossentropy(K.variable(y_true), K.variable(y_pred)))
        elif self._metric=='binary_crossentropy':
            metric = K.eval(binary_crossentropy(K.variable(y_true), K.variable(y_pred)))
        else:
            raise ValueError('Invalid value for "custom_eval_stopping["name"], "roc-auc","norm-gini","mean_squared_error", \
                             "mean_absolute_error","categorical_crossentropy","binary_crossentropy" supported.')
        return metric


    def on_train_begin(self, logs={}):
        return


    def on_train_end(self, logs={}):
        return


    def on_epoch_begin(self, epoch, logs={}):
        return
    
    
    def on_epoch_end(self, epoch, logs={}):
        """
        Log and print value for specified metric after each epoch.
        Used for early stopping.
        """
        if self.X_aux_train is not None:
            y_pred_train = self.model.predict([self.X_train,self.X_aux_train],verbose=0)
        else:
            y_pred_train = self.model.predict(self.X_train,verbose=0)
        y_pred_train = np.reshape(y_pred_train,self.y_train.shape)
        logs[self._metric + '-train'] = self._get_metric(self.y_train,y_pred_train)
        if self.X_val is not None:
            if self.X_aux_val is not None:
                y_pred_val = self.model.predict([self.X_val,self.X_aux_val],verbose=0)
            else:
                y_pred_val = self.model.predict(self.X_val,verbose=0)
            y_pred_val = np.reshape(y_pred_val,self.y_val.shape)
            logs[self._metric + '-val'] = self._get_metric(self.y_val,y_pred_val)
        
            print('\r%s-train: %s - %s-val: %s' %(self._metric,str(round(logs[self._metric + '-train'],5)),
                                                  self._metric,str(round(logs[self._metric + '-val'],5))),end=10*' '+'\n')
        else:
            print('\r%s-train: %s' %(self._metric,str(round(logs[self._metric + '-train'],5))))
        return
        
    
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
