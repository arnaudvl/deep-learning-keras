
class DNN(NN):
    
    """
    Create and train models, and make predictions for deep feedforward neural nets using keras with tensorflow backend.
    
    Main functions:
        - init_model: set up and compile convolutional neural net
        - train_model
        - predict_model
        - cv_train_predict_model: run nfold CV on neural net and make predictions on test data
    """
    
    def __init__(self,X,y,architecture,output_layer='softmax',optimizer_type='adam',loss_function='categorical_crossentropy',
                 metrics=['accuracy'],learning_rate=0.001,learning_rate_decay=0.,learning_rate_factor=0.1,num_epochs=10000,
                 early_stopping_epochs=100,custom_eval_stopping={'name':'roc-auc','mode':'max','data':'val'},nfolds=5,
                 batch_size=64,runs=3,val_size=0.2,random_state=1,scaler={'type':'minmax','columns':'all'},imputer='median',
                 dummify=[],preprocessed=False,X_pred=None,train_id=None,pred_id=None,print_out=True,print_training=True,
                 write_output=False,save_dir=None,model_name=None,target_col=['target'],id_col='id'):
        
        """
        Arguments:
            num_epochs -- int, number of passes over training set, default=10000
            nfolds -- int, number of folds for CV, default=5
            batch_size -- int, number of training samples in each step, default=64
            random_state -- int, initiate random state to generate CV folds, default=1
            X_pred -- np array, data on which predictions are made, default=None
            runs -- int, number of runs over each fold in CV with different random seed, default=3
            train_id -- np array, id's of training data, default=None
            scaler -- dict, keys 'type' and 'columns' specify the type of scaling and a list to which columns scaling is applied
                            or 'all' if applied to all columns, default={'type':'minmax','columns':'all'}
            imputer -- str, imputing applied to replace nan's in training and prediction data, default='median'
            dummify -- list, names of training columns that end with dummify will be used for OHE, default=None
            preprocessed -- bool, preprocessing will be applied if False, default=False 
            print_out -- bool, print out intermediate results of CV, default=True
            print_training -- bool, print graph with training progress over loss function, default=True
            
            See class NN for documentation of other arguments. 
        """
        
        NN.__init__(self,X,y,architecture,output_layer=output_layer,optimizer_type=optimizer_type,loss_function=loss_function,
                    metrics=metrics,learning_rate=learning_rate,learning_rate_decay=learning_rate_decay,
                    learning_rate_factor=learning_rate_factor,early_stopping_epochs=early_stopping_epochs,
                    custom_eval_stopping=custom_eval_stopping,val_size=val_size,pred_id=pred_id,
                    write_output=write_output,save_dir=save_dir,model_name=model_name,target_col=target_col,id_col=id_col)
        
        self.num_epochs = num_epochs
        self.nfolds = nfolds
        self.batch_size = batch_size
        self.runs = runs
        self.random_state = random_state
        self.scaler = scaler
        self.imputer = imputer
        self.dummify = dummify
        self.preprocessed = preprocessed
        self.X_pred = X_pred
        self.train_id = train_id
        self.print_out = print_out
        self.print_training = print_training
        
        self._X_train_test = None
        self._n_train = None
        self._cv = False
        self._print_summary = False
        
    
    def _combine_train_test(self):
        """
        Combine train and test data for scaling and dummification.
        """
        if self.X_pred is not None:
            self._n_train = self.X.shape[0]
            self._X_train_test = pd.concat((self.X,self.X_pred)).reset_index(drop=True)
        else:
            self._n_train = None
            self._X_train_test = self.X
        return self
    
    
    def _impute(self):
        """
        Impute nan's.
        """
        imp = Imputer(missing_values='NaN',strategy=self.imputer,axis=0)
        
        if self._X_train_test is None: # needs to be run after _combine_train_test()
            self._combine_train_test()
        
        X_train_test_imp = pd.DataFrame(imp.fit_transform(self._X_train_test))
        X_train_test_imp.columns = self._X_train_test.columns
        X_train_test_imp.index = self._X_train_test.index
        self._X_train_test = X_train_test_imp
        
        return self
    
    
    def _dummify(self):
        """
        Columns ending with self.dummify (str) will be dummified.
        
        Needs to be run after _combine_train_test().
        """
        if self._X_train_test is None: # needs to be run after _combine_train_test()
            self._combine_train_test()
        
        if self.dummify is not None:
            col_to_dummify = self.X.columns[self.X.columns.str.endswith(self.dummify)].astype(str).tolist()
            
            for col in col_to_dummify:
                dummy = pd.get_dummies(self._X_train_test[col].astype('category'))
                columns = dummy.columns.map(int).astype(str).tolist()
                columns = [col + '_' + w for w in columns]
                dummy.columns = columns
                self._X_train_test = pd.concat((self._X_train_test, dummy), axis=1)
            
            self._X_train_test.drop(col_to_dummify, axis=1, inplace=True)
        
        return self
    
    
    def _get_scaler(self):
        """
        Define scaler.
        """
        if self.scaler['type']=='minmax':
            self._scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.scaler['type']=='std':
            self._scaler = StandardScaler()
        else:
            raise ValueError('Only "minmax" and "std" are valid inputs for scaler["type"]')
        return self
    
    
    def _apply_scaling(self):
        """
        Scale specified columns through a pre-defined scaler or MinMaxScaler (default).
        
        Needs to be run after _combine_train_test().
        """
        if self.scaler is None:
            return self
        
        if self._X_train_test is None: # needs to be run after _combine_train_test()
            self._combine_train_test()
        
        self._get_scaler()
        
        if self.scaler['columns']=='all':
            X_train_test_scale = pd.DataFrame(self._scaler.fit_transform(self._X_train_test))
            X_train_test_scale.columns = self._X_train_test.columns
            X_train_test_scale.index = self._X_train_test.index
            self._X_train_test = X_train_test_scale
        else:
            X_train_test_scale_cols = self._scaler.fit_transform(self._X_train_test[self.scaler['columns']])
            self._X_train_test[self.scaler['columns']] = X_train_test_scale_cols
        
        return self
    
    
    def _to_categorical(self):
        """
        If >2 classes in target variable, categorize.
        """
        n_classes = self._num_classes()
        if n_classes>2 and self.y.shape[1]==1:
            self.y = to_categorical(self.y,num_classes=n_classes)
        return self
    
    
    def _to_np(self):
        """
        Convert input data from dataframe to numpy arrays used in rest of model.
        """
        if self._X_train_test is None: # needs to be run after _combine_train_test()
            self._combine_train_test()
        
        if self._n_train is None:
            self.X = self._X_train_test.values
        else:
            self.X = self._X_train_test.values[:self._n_train,:]
            self.X_pred = self._X_train_test.values[self._n_train:,:]
        return self


    def preprocess_data(self):
        """
        Data preprocessing:
        - combine train and test sets
        - impute missing values
        - apply OHE
        - scale data [-1,1]
        - transform target values to categorical data if needed
        - convert pd dataframes into numpy arrays for further use in model
        - apply train_test_split if validation data needed
        """
        self._combine_train_test()
        self._impute()
        self._dummify()
        self._apply_scaling()
        self._to_categorical()
        self._to_np()
        self.preprocessed = True
        return self
    
    
    def _set_architecture_type(self):
        """
        Set architecture type to 'fc' (fully connected) layers.
        """
        ilayer = 0
        while ilayer<len(self.architecture):
            self.architecture[ilayer]['type'] = 'fc'
            ilayer+=1
        return self
    
    
    def init_model(self):
        """
        Set up and compile deep neural net architecture.
        """
        self._set_architecture_type()
        self._get_default_values_architecture()
        ilayer = 0
        model_dnn = Sequential()
        
        if self.X_train is not None:
            input_shape = self.X_train.shape[1]
        else:
            input_shape = self.X.shape[1]
        
        # set input layer
        model_dnn.add(Dense(self.architecture[ilayer]['hidden_units'],
                            kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                            bias_initializer=self.architecture[ilayer]['bias_initializer'],
                            kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1']),
                            input_dim=input_shape))
        if self.architecture[ilayer]['batchnorm']:
            model_dnn.add(BatchNormalization())
        model_dnn.add(Activation(self.architecture[ilayer]['activation']))
        model_dnn.add(Dropout(self.architecture[ilayer]['dropout']))
        
        # add hidden layers
        while ilayer<len(self.architecture)-1:
            ilayer+=1
            
            model_dnn.add(Dense(self.architecture[ilayer]['hidden_units'],
                                kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                                bias_initializer=self.architecture[ilayer]['bias_initializer'],
                                kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1'])))
            if self.architecture[ilayer]['batchnorm']:
                model_dnn.add(BatchNormalization())
            model_dnn.add(Activation(self.architecture[ilayer]['activation']))
            model_dnn.add(Dropout(self.architecture[ilayer]['dropout']))
        
        # add output layer
        n_classes = self._num_classes()
        if self.output_layer=='sigmoid':
            if n_classes==2:
                model_dnn.add(Dense(1,activation=self.output_layer)) # output layer
            else:
                raise ValueError('sigmoid output layer suitable for binary classification, but %i classes detected.' \
                                    %(n_classes))
        else:
            model_dnn.add(Dense(n_classes,activation=self.output_layer)) # output layer
        
        # compile model
        model_dnn.compile(optimizer=self._get_optimizer(),metrics=self.metrics,loss=self.loss_function)
        
        if not self._print_summary:
            model_dnn.summary() # display model architecture
            self._print_summary = True
        
        return model_dnn
    
    
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
            self._params_fit['validation_data'] = (self.X_val,self.y_val)
            
        return self
    
    
    def train_model(self,i=0,run=0):
        """
        Train model.
        """
        if not self.preprocessed: # apply preprocessing if not already done
            self.preprocess_data()
        
        if self.val_size>0 and self.X_val is None: # need to create validation data
            self._train_test_split()
        
        self.nn_model = self.init_model() # initiate model
        
        self._get_params_fit(i=i,run=run)
        params = self._params_fit
        
        if self.X_train is not None and self.X_val is not None:
            params['x'] = self.X_train
            params['y'] = self.y_train
        else:
            params['x'] = self.X
            params['y'] = self.y
        
        self.nn_fit = self.nn_model.fit(**params)
        
        return self
    
    
    def predict_model(self,model_type='last',i=0,run=0):
        """
        Make predictions on X_pred.
        """
        if model_type=='last': # predict on best model weights
            del self.nn_model
            self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        
        if self.X_pred is not None:
            self.y_pred = self.nn_model.predict(self.X_pred,verbose=0)
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
        
        if not self.preprocessed: # apply preprocessing if not already done
            self.preprocess_data()
            
        for i, (train_index, test_index) in enumerate(skf.split(self.X,self.y)):
            self.X_train, self.X_val = self.X[train_index], self.X[test_index]
            self.y_train, self.y_val = self.y[train_index], self.y[test_index]
            train_ids, val_ids = self.train_id[train_index], self.train_id[test_index]
            
            self.init_model # set up model initialization function
            
            # repeat runs for each fold with different seed
            for run in range(self.runs):
                print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
                np.random.seed()
                
#                self.init_classifier(i=i,run=run) # initalize neural net with callbacks
                
                self.train_model(i=i,run=run) # train neural net
                
                if self.print_training: # graph with progress of model performance each run
                    self.print_training_history(i=i,run=run)
                
                # want best saved model, not last where training stopped
                # delete last model instance, load last saved checkpoint
                del self.nn_model
                self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
                scores_val_run = self.nn_model.predict(self.X_val,verbose=0)
                scores_val_run = np.reshape(scores_val_run,self.y_val.shape)
                y_pred_run = self.nn_model.predict(self.X_pred,verbose=0)
                
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
            self.save_model() # only saves best model from last run
        
        self._cv = False
        return self
