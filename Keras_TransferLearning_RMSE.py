import keras
from keras.datasets import cifar10
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
from scipy import misc
import os
import EDA_myAPI_ML as eda
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('tf')
#
# load the data
#JPG to numpy array to be use in Tensorflow/Keras
dir_oasis = "D:\ESISAR\Okayama_University\Python\Oasis_experiments_dataset\Image_Dataset\oasis\\"
dir_gaped = r"D:\ESISAR\Okayama_University\Python\Image_Dataset\GAPED_2\GAPED\GAPED\\"


csv_oasis = "D:\ESISAR\Okayama_University\Python\Oasis_experiments_dataset\Image_Dataset\oasis\OASIS.csv"
csv_gaped = [dir_gaped+"A.csv",dir_gaped+"H.csv",dir_gaped+"N.csv",
                    dir_gaped+"P.csv",dir_gaped+"Sn.csv",dir_gaped+"Sp.csv"]

# load oasis 
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = [],[],[],[],[],[],[]
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = eda.OpenCsvFile(csv_oasis)

with open(r'filenames_randorder.txt', 'r') as f:
    myNames = [line.strip() for line in f]

list_name, list_arousal, list_valence = [] , [], []
for i in range(len(myNames)):
    list_name.append(Name_oasis[int(myNames[i])-1])
    list_arousal.append(Arousal_mean_oasis[int(myNames[i])-1])
    list_valence.append(Valence_mean_oasis[int(myNames[i])-1])
# load gaped 
Name_gaped,Valence_mean_gaped,Arousal_mean_gaped =[],[],[]
for i in range(len(csv_gaped)):
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = [],[],[]
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = eda.OpenCsvFile_Gaped(csv_gaped[i])
    Name_gaped.extend(Name_gapedtmp)
    Valence_mean_gaped.extend(Valence_mean_gapedtmp)
    Arousal_mean_gaped.extend(Arousal_mean_gapedtmp)

# normalize the labels of the test set (gaped) from [0,100] to [1,7]
eda.normalize(Valence_mean_gaped) #normalize the value between [0,1] to compare with the sigmoid value
eda.normalize(Arousal_mean_gaped)

eda.normalize(list_valence)
eda.normalize(list_arousal)


#################################################################################
#
# pick train and test sets
#

dir_test = dir_gaped
dir_train = dir_oasis

#Name_trainset,Valence_mean_trainset,Valence_SD_train,Valence_N_train,Arousal_mean_train,_trainArousal_SD_train,Arousal_N_train =\
#Name_gaped,Valence_mean_gaped,Valence_SD_gaped,Valence_N_gaped,Arousal_mean_gaped,_gapedArousal_SD_gaped,Arousal_N_gaped 

if dir_train == dir_gaped:
    Name_trainset,Valence_mean_trainset,Arousal_mean_trainset = Name_gaped,Valence_mean_gaped,Arousal_mean_gaped
    Name_testset,Valence_mean_testset,Arousal_mean_testset = Name_oasis,Valence_mean_oasis,Arousal_mean_oasis
else:
    Name_trainset,Valence_mean_trainset,Arousal_mean_trainset = list_name,list_valence,list_arousal
    Name_testset,Valence_mean_testset,Arousal_mean_testset = Name_gaped,Valence_mean_gaped,Arousal_mean_gaped
percentage = 80
NTrain = int(len(Name_trainset)*(percentage/100))
NTest = len(Name_testset) #int(len(Name_test)*(percentage/100))
TestResultList = []

#Valence_mean_trainset = eda.normalize(Valence_mean_trainset)



# load inceptionV3 model + remove final classification layers
model = InceptionV3(weights='imagenet', include_top=False, input_shape=(500, 400, 3))
#model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
#model = keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
print('model loaded')
for i in range(1):
    Name_shuffled_trainset = np.array(Name_trainset)
    Arousal_mean_trainset_shuffled = np.array(Arousal_mean_trainset)

#    perm = np.random.permutation(len(Name_trainset))
#    np.take(Name_trainset,perm,axis=0,out=Name_shuffled_trainset)
#    np.take(Arousal_mean_trainset,perm,axis=0,out=Arousal_mean_trainset_shuffled)
    
    
    # no need to shuffle test set
    # Name_shuffled_test = np.array(Name_test)
    # Valence_mean_testset_shuffled = np.array(Valence_mean_testset)
    # perm = np.random.permutation(len(Name_test))
    # np.take(Name_test,perm,axis=0,out=Name_shuffled_test)
    # np.take(Valence_mean_testset,perm,axis=0,out=Valence_mean_testset_shuffled)
    

    Name_trainset_trainsubset, Name_trainset_testsubset = Name_shuffled_trainset[:NTrain],Name_shuffled_trainset[NTrain:]    
    Label_trainset_trainsubset, Label_trainset_testsubset = Arousal_mean_trainset_shuffled[:NTrain],Arousal_mean_trainset_shuffled[NTrain:]
    Label_testset = Arousal_mean_testset
    
    x_train = []
    x_test_sameset = []
    x_test_crossset = []

    

    for i in range(len(Name_trainset_trainsubset)):
        if dir_train == dir_gaped:
           subdir_trainsubset = Name_trainset_trainsubset[i][0]
        else :
           subdir_trainsubset = 'F'
        im = scipy.misc.imread(dir_train + subdir_trainsubset+"/"+Name_trainset_trainsubset[i]+".jpg",False,mode='RGB')
        x_train.append(im)
    
    for i in range(len(Name_trainset_testsubset)):
        if dir_train == dir_gaped:
            subdir_testsubset = Name_trainset_testsubset[i][0]
        else :
            subdir_testsubset = 'F'
        im = scipy.misc.imread(dir_train + subdir_testsubset+"/"+Name_trainset_testsubset[i]+".jpg",False,mode='RGB')
        x_test_sameset.append(im)

    for i in range(len(Name_testset)):
        if dir_test == dir_gaped:
            subdir_test = Name_testset[i][0]
        else :
            subdir_test = 'F'
        im = scipy.misc.imread(dir_test+subdir_test+"/"+Name_testset[i]+".jpeg",False,mode='RGB')
        x_test_crossset.append(im)
    
    x_train = np.array(x_train)
    x_test_sameset = np.array(x_test_sameset)
    x_test_crossset = np.array(x_test_crossset)

    y_train = np.array(Label_trainset_trainsubset)
    y_test_sameset = np.array(Label_trainset_testsubset)
    y_test_crossset = np.array(Label_testset)
   ################################################################################# 
	#
	# For loading and serializing the image
	# img_path = 'elephant.jpg'
	# img = image.load_img(img_path, target_size=(224, 224))
	# x = image.img_to_array(img)
	# x = np.expand_dims(x, axis=0)
	# x = preprocess_input(x)
	#
	#################################################################################

    y_train = np.squeeze(y_train)
    print('Train/Test data loaded')
    #################################################################################
    # obtain bottleneck features (train)
    if os.path.exists('inception_features_train.npz'):
        print('bottleneck features detected (train)')
        features_train = np.load('inception_features_train.npz')['features_train']
    else:
        print('bottleneck features file not detected (train)')
        print('calculating now ...')
        # pre-process the train data
        big_x_train = np.array([scipy.misc.imresize(x_train[i], (500, 400, 3)) 
                                for i in range(0, len(x_train))]).astype('float32')
#        big_x_train = tf.contrib.keras.preprocessing.image.img_to_array(x_train[i] for i in range(0, len(x_train)))
#        big_x_train = np.expand_dims(big_x_train, axis=0)
#        
        inception_input_train = preprocess_input(big_x_train)
        print('train data preprocessed')
        # extract, process, and save bottleneck features
        features_train = model.predict(inception_input_train)
        features_train = np.squeeze(features_train)
        np.savez('inception_features_train', features_train=features_train)
    print('bottleneck features saved (train)')
    #################################################################################
    # obtain bottleneck features (test_sameset)
    if os.path.exists('inception_features_test_sameset.npz'):
        print('bottleneck features detected (test_sameset)')
        features_test_sameset = np.load('inception_features_test_sameset.npz')['features_test_sameset']
    else:
        print('bottleneck features file not detected (test_sameset)')
        print('calculating now ...')
        # pre-process the test data
        big_x_test_sameset = np.array([scipy.misc.imresize(x_test_sameset[i], (500, 400, 3)) 
                           for i in range(0, len(x_test_sameset))]).astype('float32')

#        big_x_test_sameset = tf.contrib.keras.preprocessing.image.img_to_array(x_test_sameset[i] for i in range(0, len(x_test_sameset))).astype('float32')
#        big_x_test_sameset = np.expand_dims(big_x_test_sameset, axis=0)
        
        inception_input_test_sameset = preprocess_input(big_x_test_sameset)
        # extract, process, and save bottleneck features (test_sameset)
        features_test_sameset = model.predict(inception_input_test_sameset)
        features_test_sameset = np.squeeze(features_test_sameset)
        np.savez('inception_features_test_sameset', features_test_sameset=features_test_sameset)
    print('bottleneck features saved (test_sameset)')
   #################################################################################
    # obtain bottleneck features (test_crossset)
    if os.path.exists('inception_features_test_crossset.npz'):
        print('bottleneck features detected (test_crossset)')
        features_test_crossset = np.load('inception_features_test_crossset.npz')['features_test_crossset']
    else:
        print('bottleneck features file not detected (test_crossset)')
        print('calculating now ...')
        # pre-process the test data
        big_x_test_crossset = np.array([scipy.misc.imresize(x_test_crossset[i], (500, 400, 3)) 
                           for i in range(0, len(x_test_crossset))]).astype('float32')

#        big_x_test_crossset = tf.contrib.keras.preprocessing.image.img_to_array(x_test_crossset[i] for i in range(0, len(x_test_crossset))).astype('float32')
#        big_x_test_crossset = np.expand_dims(big_x_test_crossset, axis=0)
        
        inception_input_test_crossset = preprocess_input(big_x_test_crossset)
        # extract, process, and save bottleneck features (test_crossset)
        features_test_crossset = model.predict(inception_input_test_crossset)
        features_test_crossset = np.squeeze(features_test_crossset)
        np.savez('inception_features_test_crossset', features_test_crossset=features_test_crossset)
    print('bottleneck features saved (test_crossset)')
    #################################################################################
    from keras.utils import np_utils
    
    # one-hot encode the labels
#    y_train = np_utils.to_categorical(y_train, 1024)
#    y_test = np_utils.to_categorical(y_test, 1024)
    
    from keras.callbacks import ModelCheckpoint   
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D,Flatten,MaxPooling2D,Activation


# =============================================================================
#     model = Sequential()
#     model.add(Conv2D(filters=512, kernel_size=2, input_shape=(3, 3, 2048) ))
# #    model.add(Dropout(0.8))
#     model.add(GlobalAveragePooling2D())
# #    model.add(Dense(512, activation='relu'))
# #    model.add(Dropout(0.6))
# #    model.add(Dense(256, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.summary()
# =============================================================================
   
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, input_shape=features_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='Root_mean_squared_error',optimizer= 'adam',
                  metrics= ['Root_mean_squared_error'])
#    model.compile(loss='mean_squared_error',optimizer='adam', metrics= ['mean_squared_error'])
    callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto'),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint(filepath='model.best.hdf5', 
                                   verbose=1, save_best_only=True),
]
    model.fit(features_train, y_train, batch_size=18, epochs=300,
              validation_split=0.2, callbacks=callbacks,
              verbose=2, shuffle=True)

    # use tensorboard for displaying the real time evolition of error

    # load the weights that yielded the best validation accuracy
    model.load_weights('model.best.hdf5')
    
    # evaluate test accuracy
    score_sameset = model.evaluate(features_test_sameset, y_test_sameset, verbose=0)
    accuracy_sameset = score_sameset[1]

    score_crossset = model.evaluate(features_test_crossset, y_test_crossset, verbose=0)
    accuracy_crossset = score_crossset[1]
    TestResultList.append(((7-1)*accuracy_sameset)+1)
    TestResultList.append(((7-1)*accuracy_crossset)+1)
    # print test accuracy
    print('Test accuracy (same set): %f' % accuracy_sameset)
    print('Test accuracy (cross set): %f' % accuracy_crossset)


print(TestResultList)

#Test accuracy (same set): 2.657460
#Test accuracy (cross set): 2.902688
#[2.6574599207117315, 2.9026880094449816]