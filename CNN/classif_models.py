import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import os.path as op
import gc
import pickle
from sklearn.pipeline import make_pipeline
#import utils.utils_tda as utda
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
seed = 0
tf.keras.utils.set_random_seed(
    seed
)
def plot_epochs_metric(hist, file_name, metric='loss'):
    print(hist.history)
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def load_obj(name, path):
    with open(op.join(path, name), 'rb') as f:
        return pickle.load(f)

########################################################################### features ###########################################################################
def compute_window_ppa(window):
    return np.max(window, axis=0) - np.min(window, axis=0)

def compute_window_upslope(window):
    return np.max(np.diff(window, axis=0), axis=0)

def compute_window_std(window):
    return np.std(window, axis=0)

def compute_window_average_slope(window):
    abs_slopes = np.abs(np.diff(window, axis=0))
    return np.max((abs_slopes[:-1] + abs_slopes[1:]) / 2)

list_of_features = ["ppa", "upslope", "std", "average_slope"]
compute_features = {"ppa" : compute_window_ppa, "upslope" : compute_window_upslope, "std": compute_window_std, "average_slope": compute_window_average_slope}
########################################################################### data generation ###########################################################################

# Data generator to load images batch per batch
class DataGenerator_memeff(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, dim, shuffle, path, aug):
    
        'Initialization'
        self.dim = dim # dimension of the window
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle # bool if want to shuffle output windows
        self.path = path # where the .pkl and binary files are
        self.aug = aug # bool if want to apply augmentation
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp = np.array(list_IDs_temp)
        list_IDs_temp = list_IDs_temp[list_IDs_temp[:, 1].argsort()].tolist()

        # Generate data
        (X, features), y = self.__data_generation(list_IDs_temp)

        return (X, features), y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
       # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization   
        X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1])) # contains window data
        features_batch = np.empty((self.batch_size, len(list_of_features), self.dim[1]))
        y_batch = np.empty((self.batch_size), dtype=int) # contains window labels
        
        if len(self.dim) == 3:
            X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1], 1)) # contains images

        # Generate data
        prevsub = 1000

        for i, ID in enumerate(list_IDs_temp):
            # get infos from the "ids" array
            win = np.array(ID)[0]
            sub = np.array(ID)[1]
            label = np.array(ID)[2]
            # if the subject is different from the previous one, then open the subject file
            if(sub != prevsub): 
                f = open(op.join(self.path, 'data_raw_'+str(sub).zfill(3)+'_b3_windows_bi'))
            # Read the window data from the binary file (seek=move the cursor, fromfile=extract the correct nb of bytes) 
            f.seek(self.dim[0]*self.dim[1]*win*4) #4 because its float32 and dtype.itemsize = 4
            sample = np.fromfile(f, dtype='float32', count=self.dim[0]*self.dim[1])
            # reshape the window data
            sample = sample.reshape(self.dim[1],self.dim[0])
            sample = np.swapaxes(sample,0,1)
            # add a "channel" dimension needed for a CNN (last dim in tensorflow)
            if len(self.dim) == 3:
                sample = np.expand_dims(sample,axis=-1)

            for feat, func in compute_features.items():
                features_batch[i,list_of_features.index(feat)] = func(sample)
                
            if (self.aug):
                #Apply gaussian noise as augmentation in half of the samples
                if np.random.uniform() >= .5:
                    #augmentation - add gaussian noise
                    noise = np.random.normal(0,0.01, size=self.dim)
                    sample_noise = sample+noise
                    #standardize again as data were already standardized (not sure this is necessary)
                    # mean = np.mean(sample_noise)
                    # std = np.std(sample_noise)
                    # sample_augmented = (sample_noise - mean)/std
                    sample_augmented = sample_noise
                else:
                    sample_augmented = sample
            else:
                sample_augmented = sample

            X_batch[i,] = sample_augmented

            # Store classf
            y_batch[i,] = label
            prevsub = sub

        #shuffle        
        X_batch, features_batch, y_batch = shuffle(X_batch, features_batch, y_batch, random_state=0)
        return (X_batch, features_batch), y_batch

def load_generators_memeff(X_train_ids, X_valid_ids, X_test_ids, aug, window_size, sfreq, path_extracted_data, batch_size):
    # X_train_ids, X_valid_ids, X_test_ids: id arrays obtained from the function generate_database_with_hold_out_memeff
    # aug: if want to apply augmentation.
    # window_size, sfreq: params used to create the windows
    # path_extracted_data: where the .pkl and binary files are
    # returns data generators

    dim = (int(sfreq*window_size), 274)

    training_generator = DataGenerator_memeff(X_train_ids.tolist(), batch_size, dim, True, path_extracted_data, aug) #last arg true if want augmentation  
    #Never use augmentation for valid and test sets 
    valid_generator = DataGenerator_memeff(X_valid_ids.tolist(), batch_size, dim, True, path_extracted_data, False)
    testing_generator = DataGenerator_memeff(X_test_ids.tolist(), 1, dim, False, path_extracted_data, False)

    return training_generator, valid_generator, testing_generator

########################################################################### metrics ###########################################################################

def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    print(y_true)
    print(y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

########################################################################### CNN model ###########################################################################

def classify_sfcn(first_direction,sfreq,window_size,initial_bias):

    reg = 0
    if first_direction == 'channel':
        input_shape = (274, int(sfreq*window_size), 1)
    elif first_direction == 'time':
        input_shape = (int(sfreq*window_size), 274, 1)

    input_layer = keras.layers.Input(input_shape)
    input_features = keras.layers.Input(shape=(len(list_of_features), input_shape[1]))

    #block 1
    x=keras.layers.Conv2D(filters=32, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(input_layer)
    x=keras.layers.BatchNormalization(name="t1_norm1")(x)

    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 2
    x=keras.layers.Conv2D(filters=64, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 3
    x=keras.layers.Conv2D(filters=128, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 4
    x=keras.layers.Conv2D(filters=256, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 6
    x=keras.layers.Conv2D(filters=64, kernel_size=(1, 1),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)

    #block 7, different from paper
    x=keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x=keras.layers.Dropout(.5)(x)
    x=keras.layers.Flatten()(x)
    
    output_layer = keras.layers.Dense(1, activation='sigmoid',bias_initializer=tf.constant_initializer(initial_bias))(x)

    model = keras.models.Model(inputs=[input_layer, input_features], outputs=output_layer)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                                      min_lr=0.00001)
    
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False,gamma=2), optimizer=keras.optimizers.Adam(0.0001),
                      metrics=['accuracy','Precision','Recall'])


    return model, reduce_lr

def classify_sfcn_with_features(first_direction,sfreq,window_size,initial_bias):

    reg = 0
    if first_direction == 'channel':
        input_shape = (274, int(sfreq*window_size), 1)
    elif first_direction == 'time':
        input_shape = (int(sfreq*window_size), 274, 1)

    input_layer = keras.layers.Input(input_shape)
    input_features = keras.layers.Input(shape=(len(list_of_features), input_shape[1]))

    #block 1
    x=keras.layers.Conv2D(filters=32, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(input_layer)
    x=keras.layers.BatchNormalization(name="t1_norm1")(x)

    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 2
    x=keras.layers.Conv2D(filters=64, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 3
    x=keras.layers.Conv2D(filters=128, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 4
    x=keras.layers.Conv2D(filters=256, kernel_size=(5, 5),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2))(x)
    x=keras.layers.LeakyReLU()(x)

    #block 6
    x=keras.layers.Conv2D(filters=64, kernel_size=(1, 1),padding='same',kernel_regularizer=keras.regularizers.l2(reg))(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.LeakyReLU()(x)

    #block 7, different from paper
    x=keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    x=keras.layers.Dropout(.5)(x)
    x=keras.layers.Flatten()(x)

    x_features=keras.layers.Flatten()(input_features)
    
    x=keras.layers.Concatenate()([x,x_features])
    
    output_layer = keras.layers.Dense(1, activation='sigmoid',bias_initializer=tf.constant_initializer(initial_bias))(x)

    model = keras.models.Model(inputs=[input_layer, input_features], outputs=output_layer)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                                      min_lr=0.00001)
    
    model.compile(loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False,gamma=2), optimizer=keras.optimizers.Adam(0.0001),
                      metrics=['accuracy','Precision','Recall'])


    return model, reduce_lr

def fit_model(model, file_path, training_generator, valid_generator, callbacks, class_weight, batch_size, nb_epochs):
    # model: instantiated model
    # file_path: where to save the model file and results
    # window_size, sfreq: params used to create the windows
    # training_generator, valid_generator: training and validation data generators
    # callbacks: callbacks (only model checkpoint used so far)
    # class_weight: class_weight as computed when preparing the data
    # returns data generators

    # saves the best model while monitoring the validation loss

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path+'_best_model.keras',
                                                           monitor='val_loss', save_best_only=False,verbose=1)
    if callbacks == None:
        my_callbacks =model_checkpoint
    else:
        my_callbacks =[model_checkpoint,callbacks]
    hist = model.fit(training_generator, validation_data=valid_generator, batch_size=batch_size, epochs=nb_epochs, callbacks = my_callbacks, class_weight=class_weight)
    
    # generate output plot showing the f1_score for the spike class over training epochs  
    plot_epochs_metric(hist, file_path+'_loss_results.png', metric='loss')
    plot_epochs_metric(hist, file_path+'_Precision_results.png', metric='precision')
    plot_epochs_metric(hist, file_path+'_Recall_results.png', metric='recall')
    # plt.plot(hist.history['val_Precision'])
    # plt.savefig(file_path+'_Precision_results.png')
    # plt.plot(hist.history['val_Recall'])
    # plt.savefig(file_path+'_Recall_results.png')

    del model
    del hist
    gc.collect()
    keras.backend.clear_session()

def test_model(file_path, model_type, testing_generator, X_test_ids, path_extracted_data):
    # file_path: where the model is saved
    # testing_generator: testing data generators
    # X_test_ids: "ids" array of the test data
    # path_extracted_data: where the .pkl and binary files are


    # Load the model
    model = keras.models.load_model(file_path+'_best_model.keras', custom_objects={"f1_m": f1_m })
    # get predictions
    y_pred_probas = model.predict(testing_generator)
    # retrieve true labels from the "ids" array 
    y_test = X_test_ids[:,2]

    # Plot roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path+'_roc.png')
    # threshold predictions at .5
    y_pred = (y_pred_probas > 0.5).astype("int32")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)

    # save result metrics in csv
    fields=[accuracy_score(y_test,y_pred),f1_score(y_test,y_pred),specificity,recall_score(y_test,y_pred),confusion_matrix(y_test, y_pred).ravel()]
    print(fields)
    add_header=False
    if not (os.path.exists(file_path+'_cv_resuls_'+model_type+'.csv')):
        add_header=True

    with open(file_path+'_cv_results_'+model_type+'.csv', 'a', newline='') as f:
        writer = csv.writer(f)           
        if add_header:
            writer.writerow(["accuracy","f1-score","specificity","sensitivity","confusion matrix"])
        writer.writerow(fields)

    # save predictions + subject id, window id, and true label in csv
    sub = X_test_ids[:,1]
    win = X_test_ids[:,0]

    prevsub = 1000
    # to do so, go read the timing and block files to integrate that info in the csv
    for ind, i in enumerate(sub):
        if (i != prevsub):
            y_timing_data = load_obj('data_raw_'+str(i).zfill(3)+'_b3_timing.pkl',path_extracted_data)
            y_block_data = load_obj('data_raw_'+str(i).zfill(3)+'_b3_blocks.pkl',path_extracted_data)

        y_timing = y_timing_data[win[ind]]
        y_block = y_block_data[win[ind]]

        add_header=False
        if not (os.path.exists(file_path+'_cvpredictions_'+model_type+'.csv')):
            add_header=True

        with open(file_path+'_cvpredictions_'+model_type+'.csv', 'a', newline='') as f:
            writer = csv.writer(f)           
            if add_header:
                writer.writerow(["subject","block","timing","test","pred"])
            writer.writerow([i,y_block,y_timing,y_test[ind],y_pred[ind]])

        prevsub=i

    keras.backend.clear_session()

    return y_pred_probas
