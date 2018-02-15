
class CNNTransferLearning(CNN):
    
    """
    Create convolutional neural nets using transfer learning from keras with tensorflow backend.
    Good resource on best practice: http://cs231n.github.io/transfer-learning/
    
    Images should have min dimension for each architecture (https://keras.io/applications/):
        - VGG16: 48x48x3
        - VGG19: 48x48x3
        - ResNet50: 197x197x3
        - Xception: 71x71x3
        - InceptionV3: 139x139x3
        - InceptionResNetV2: 139x139x3
        - MobileNet: 32x32x3
        
    Other functionality from CNN.
    """
    
    def __init__(self,X,y,architecture,base_model,X_aux=None,trainable_layers=None,remove_from_layer=None,
                 output_layer='softmax',optimizer_type='adam',loss_function='categorical_crossentropy',
                 metrics=['accuracy'],learning_rate=0.001,learning_rate_decay=0.,learning_rate_factor=0.1,
                 num_epochs=10000,early_stopping_epochs=100,custom_eval_stopping={'name':'roc-auc','mode':'max','data':'val'},
                 nfolds=5,batch_size=64,pooling='max',runs=3,val_size=0.2,trainable_layers_tune=None,
                 remove_from_layer_tune=None,load_wgt=False,random_state=1,X_pred=None,X_aux_pred=None,
                 train_id=None,pred_id=None,scale_data=None,augment_data={},scale_pre_concat=False,print_out=True,
                 print_training=True,write_output=False,save_dir=None,model_name=None,target_col=['target'],id_col='id'):
        
        """
        Arguments:
            base_model -- str, model used for transfer learning
                               options: 'VGG16','VGG19','ResNet50','Xception','InceptionV3','InceptionResNetV2','MobileNet'
            trainable_layers -- list, names of layers base model to retrain weights from, default=None
            remove_from_layer -- str, name of layer from which the base model's layers are removed, including 'remove_from_layer',
                                      eg 'block5_conv1' will remove the 'block5_conv1' until 'block5_pool' layers and add the
                                      pooling/flattening and fully connected layers defined in the architecture on top,
                                      default=None
            trainable_layers_tune -- list, names of layers base model to retrain weights from during model tuning, default=None
            remove_from_layer_tune -- list, names of layers from where base model's layers are removed during tuning, default=None
            
            See class CNN for documentation of other arguments. 
        """
        
        CNN.__init__(self,X,y,architecture,X_aux=X_aux,output_layer=output_layer,optimizer_type=optimizer_type,
                     loss_function=loss_function,metrics=metrics,learning_rate=learning_rate,learning_rate_decay=learning_rate_decay,
                     learning_rate_factor=learning_rate_factor,num_epochs=num_epochs,early_stopping_epochs=early_stopping_epochs,
                     custom_eval_stopping=custom_eval_stopping,nfolds=nfolds,batch_size=batch_size,pooling=pooling,runs=runs,
                     val_size=val_size,random_state=random_state,X_pred=X_pred,X_aux_pred=X_aux_pred,train_id=train_id,
                     pred_id=pred_id,scale_data=scale_data,augment_data={},scale_pre_concat=scale_pre_concat,load_wgt=load_wgt,
                     print_out=print_out,print_training=print_training,write_output=write_output,save_dir=save_dir,
                     model_name=model_name,target_col=target_col,id_col=id_col)
                     
        self.base_model = base_model
        self.trainable_layers = trainable_layers
        self.remove_from_layer = remove_from_layer
        self.trainable_layers_tune = trainable_layers_tune
        self.remove_from_layer_tune = remove_from_layer_tune
        
        if self._mode=='min':
            self._score_mult = -1
        else:
            self._score_mult = 1
        
        self._tune_score = None
        self._best_overall_score = False
    
    
    def _get_base_model(self):
        """
        Define base model used in transfer learning.
        """
        if self.base_model=='VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='VGG19':
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='Xception':
            base_model = Xception(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='InceptionResNetV2':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        elif self.base_model=='MobileNet':
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=self.X.shape[1:])
        else:
            raise ValueError('Valid base model values are: "VGG16","VGG19","ResNet50","Xception","InceptionV3", \
                             "InceptionResNetV2","MobileNet".')
        return base_model
    
    
    def init_model(self):
        """
        Set up and compile convolutional neural net architecture.
        Overrides init_model in CNN class.
        """
        self._get_default_values_architecture()
        base_model = self._get_base_model()
        
        # remove undesired layers
        if self.remove_from_layer is None:
            x = base_model.output
        else:
            remove_layer = 0
            for layer in base_model.layers:
                if layer.name==self.remove_from_layer:
                    break
                remove_layer+=1
            x = base_model.layers[remove_layer-1].output
        
        # pooling/flattening method
        if self.pooling=='avg':
            x = GlobalAveragePooling2D()(x)
        elif self.pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif self.pooling=='flat':
            x = Flatten()(x)
        else:
            raise ValueError('Only "max", "avg" and "flat" are valid flattening/pooling methods.')
            
        # add batchnorm option in case auxiliary input is scaled
        if self.X_aux is not None and self.scale_pre_concat:
            x = BatchNormalization()(x)
        
        # merge model with auxiliary input
        if self.X_aux is not None:
            if self.X_aux.shape[1:]==():
                aux_shape = (1,)
            else:
                aux_shape = self.X_aux.shape[1:]
            aux_input = Input(shape=aux_shape, name='aux_input')
            x = concatenate([x,aux_input])
        
        # add fully connected layers
        ilayer = 0
        while ilayer<len(self.architecture):
            x = Dense(self.architecture[ilayer]['hidden_units'],
                      kernel_initializer=self.architecture[ilayer]['kernel_initializer'],
                      bias_initializer=self.architecture[ilayer]['bias_initializer'],
                      kernel_regularizer=l1(self.architecture[ilayer]['kernel_regularizer_l1']),
                      name='fc_' + str(ilayer+1))(x)
            if self.architecture[ilayer]['batchnorm']:
                x = BatchNormalization()(x)
            x = Activation(self.architecture[ilayer]['activation'])(x)
            x = Dropout(self.architecture[ilayer]['dropout'])(x)
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
        
        # freeze all base model layers unless specified otherwise
        for layer in base_model.layers:
            layer.trainable = False
        
        if self.trainable_layers is not None:
            for layer in base_model.layers:
                if layer.name in self.trainable_layers:
                    layer.trainable = True
        
        # add auxiliary input
        if self.X_aux is not None:
            model_input = [base_model.input,aux_input]
        else:
            model_input = base_model.input
        
        # create model
        model_cnn_tl = Model(inputs=model_input,outputs=predictions)
        
        # load weights if specified
        if self.load_wgt:
            model_cnn_tl.load_weights(self.model_name + '_weights.h5')
        
        # compile model
        model_cnn_tl.compile(optimizer=self._get_optimizer(),metrics=self.metrics,loss=self.loss_function)
        
        if not self._print_summary:
            model_cnn_tl.summary() # display model architecture
            self._print_summary = True
        
        return model_cnn_tl
    
    
    def _load_model_and_calc_metric(self,i=0):
        """
        Load best model from last checkpoint, make out-of-fold predictions and scores.
        Check if score is better than overall score so far and override best overall model if so.
        """
        self._best_overall_score = False
        
        # want best saved model, not last where training stopped
        # delete last model instance, load last saved checkpoint
        del self.nn_model
        self.nn_model = load_model(self.model_name + '-fold-' + str('%02d' % (i + 1)) + '-run-01.check')
        if self.X_aux_val is not None:
            scores_val_run = self.nn_model.predict([self.X_val,self.X_aux_val],verbose=0)
        else:
            scores_val_run = self.nn_model.predict(self.X_val,verbose=0)
        scores_val_run = np.reshape(scores_val_run,self.y_val.shape)
        
        # compare score latest model with best model so far and store best overall model
        score_layer = self._calc_metric(self.y_val,scores_val_run)
        print(' %s score: %.5f' % (self._val_loss,score_layer))
            
        if not self._tune_score:
            print(' Overall best score %s score: %.5f\n' % (self._val_loss,score_layer))
            self._tune_score = score_layer
            self.save_model(model_type='best')
            self._best_overall_score = True
        elif (score_layer * self._score_mult) > (self._tune_score * self._score_mult):
            print(' Overall best score %s score improved from %.5f to %.5f\n' % (self._val_loss,self._tune_score,score_layer))
            self._tune_score = score_layer
            self.save_model(model_type='best')
            self._best_overall_score = True
        else:
            print(' Overall best score %s of %.5f not improved\n' % (self._val_loss,self._tune_score))
                    
        return self
    
    
    def tune_layers_model(self):
        """
        Optimize transfer learning model by checking which layers to keep.
        """
        self._tune_score = None
        
        # loop over the layers to be removed and set remove_from_layer variable
        i = 0
        remove_from_layer_temp = self.remove_from_layer
        
        for remove_from_layer in self.remove_from_layer_tune:
            
            self._print_summary = False # print model summary each iteration
        
            # train model with relevant layers removed and store best model each removal option
            self.remove_from_layer = remove_from_layer
            self.train_model(i=i)
            
            if self.print_training: # graph with progress of model performance
                self.print_training_history(i=i)
            
            print('\n Remove base model %s layers from layer %s onwards...' % (self.base_model,remove_from_layer))
            
            self._load_model_and_calc_metric(i=i)
            
            if self._best_overall_score:
                best_remove_layer = remove_from_layer
                    
            i+=1
        
        print('\n Best remove layer for base model %s: %s' % (self.base_model,best_remove_layer))
        print('\n Best %s score: %.5f\n' % (self._val_loss,self._tune_score))
        
        self.remove_from_layer = remove_from_layer_temp
        
        return self
    
    
    def tune_weights_model(self):
        """
        Optimize transfer learning model by checking which layers to retrain.
        """
        self._tune_score = None
        _base_model = self._get_base_model()
        
        # freeze layers base model, train with fully connected layers in architecture
        # save weights from best checkpoint
        trainable_layers_temp = self.trainable_layers
        trainable_layers_tune_temps = self.trainable_layers_tune
        self.trainable_layers = None
        self.train_model()
        self.save_model_weights(model_type='last')
        
        self.load_wgt = True # make sure to load weights of previously trained model
        
        self.learning_rate = self.learning_rate / 10 # reduce learning rate for fine tuning
        
        # iteratively unfreeze a layer and retrain until no further improvement is reached
        i = 0
        self.trainable_layers = []
        n = len(self.trainable_layers_tune)
        
        while i<n:
            
            self._print_summary = False # print model summary each iteration
            
            # set trainable layers
            for layer in reversed(_base_model.layers):
                if layer.name in self.trainable_layers_tune:
                    self.trainable_layers.append(layer.name)
                    self.trainable_layers_tune.remove(layer.name)
                    new_retrain_layer = layer.name
                    break
            
            self.train_model(i=i)
            
            if self.print_training: # graph with progress of model performance
                self.print_training_history(i=i)
            
            print('\n Retrain base model %s layers %s...' % (self.base_model,self.trainable_layers))
            
            self._load_model_and_calc_metric(i=i)
            
            if self._best_overall_score:
                self.save_model_weights(model_type='best',i=i)
            else:
                print('\n Stop training at %s because training %s does not improve performance\n' 
                      % (self.trainable_layers,new_retrain_layer))
                self.trainable_layers = self.trainable_layers[:-1]
                break # stop iterations if retraining additional layer does not help
                    
            i+=1
        
        print('\n Best layers to retrain from base model %s: %s' % (self.base_model,self.trainable_layers))
        print('\n Best %s score: %.5f\n' % (self._val_loss,self._tune_score))
        
        self.trainable_layers = trainable_layers_temp
        self.trainable_layers_tune = trainable_layers_tune_temps
        
        return self
