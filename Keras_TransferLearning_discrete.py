import keras
from keras.datasets import cifar10
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import scipy
from scipy import misc
import os
import EDA_myAPI_ML as eda
from keras.preprocessing.image import ImageDataGenerator
from shutil import copy


img_height = 400
img_width = 500
#
# load the data
#JPG to numpy array to be use in Tensorflow/Keras
dir_oasis = "D:\ESISAR\Okayama_University\Python\Image_Dataset\oasis\\"
dir_gaped = r"D:\ESISAR\Okayama_University\Python\Image_Dataset\GAPED_2\GAPED\GAPED\\"


csv_oasis = "D:\ESISAR\Okayama_University\Python\Image_Dataset\oasis\OASIS.csv"
csv_gaped = [dir_gaped+"A.csv",dir_gaped+"H.csv",dir_gaped+"N.csv",
                    dir_gaped+"P.csv",dir_gaped+"Sn.csv",dir_gaped+"Sp.csv"]

# load oasis 
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = [],[],[],[],[],[],[]
Name_oasis,Valence_mean_oasis,Valence_SD_oasis,Valence_N_oasis,Arousal_mean_oasis,_oasisArousal_SD_oasis,Arousal_N_oasis = eda.OpenCsvFile(csv_oasis)

# load gaped 
Name_gaped,Valence_mean_gaped,Arousal_mean_gaped =[],[],[]
for i in range(len(csv_gaped)):
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = [],[],[]
    Name_gapedtmp,Valence_mean_gapedtmp,Arousal_mean_gapedtmp = eda.OpenCsvFile_Gaped(csv_gaped[i])
    Name_gaped.extend(Name_gapedtmp)
    Valence_mean_gaped.extend(Valence_mean_gapedtmp)
    Arousal_mean_gaped.extend(Arousal_mean_gapedtmp)

# normalize the labels of the test set (gaped) from [0,100] to [1,7]
eda.normalizerange(Valence_mean_gaped,1,7)
eda.normalizerange(Arousal_mean_gaped,1,7)

eda.rescaling(Valence_mean_gaped,1,7)
eda.rescaling(Arousal_mean_gaped,1,7)
eda.rescaling(Valence_mean_oasis,1,7)
eda.rescaling(Arousal_mean_oasis,1,7)


#################################################################################
#
# pick train and test sets
#
dir_train = dir_gaped
dir_test = dir_oasis


#Name_trainset,Valence_mean_trainset,Valence_SD_train,Valence_N_train,Arousal_mean_train,_trainArousal_SD_train,Arousal_N_train =\
#Name_gaped,Valence_mean_gaped,Valence_SD_gaped,Valence_N_gaped,Arousal_mean_gaped,_gapedArousal_SD_gaped,Arousal_N_gaped 

if dir_train == dir_gaped:
    Name_trainset,Valence_mean_trainset,Arousal_mean_trainset = Name_gaped,Valence_mean_gaped,Arousal_mean_gaped
    Name_testset,Valence_mean_testset,Arousal_mean_testset = Name_oasis,Valence_mean_oasis,Arousal_mean_oasis
else:
    Name_trainset,Valence_mean_trainset,Arousal_mean_trainset = Name_oasis,Valence_mean_oasis,Arousal_mean_oasis
    Name_testset,Valence_mean_testset,Arousal_mean_testset = Name_gaped,Valence_mean_gaped,Arousal_mean_gaped
    


percentage = 80
percentageValidation = 20
NTrain = int(len(Name_trainset)*(percentage/100))
NTest_subset = int(len(Name_trainset)*(100-percentage/100))
NValidation = int(NTrain * (percentageValidation/100))
NTest = len(Name_testset) #int(len(Name_test)*(percentage/100))
TestResultList = []
batch_size = 4

train_data_dir = "D:/ESISAR/Okayama_University/Python/Keras/images/train/"
validation_data_dir = "D:/ESISAR/Okayama_University/Python/Keras/images/validation/"
test_sameset_data_dir = "D:/ESISAR/Okayama_University/Python/Keras/images/testsubset/"
test_crossset_data_dir = "D:/ESISAR/Okayama_University/Python/Keras/images/testcrossset/"
#Valence_mean_trainset = eda.normalize(Valence_mean_trainset)



# load inceptionV3 model + remove final classification layers
model = InceptionV3(weights='imagenet', include_top=False, input_shape=(400, 500, 3))
#model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
#model = keras.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
print('model loaded')
for i in range(3):
    #Create directory if they don't existe
    #Empty the directory if they exist
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
        os.makedirs(train_data_dir+"negative")
        os.makedirs(train_data_dir+"positive")
        os.makedirs(train_data_dir+"null")
    if not os.path.exists(validation_data_dir):
        os.makedirs(validation_data_dir)
        os.makedirs(validation_data_dir+"negative")
        os.makedirs(validation_data_dir+"positive")
        os.makedirs(validation_data_dir+"null")
    if not os.path.exists(test_sameset_data_dir):
        os.makedirs(test_sameset_data_dir)
        os.makedirs(test_sameset_data_dir+"negative")
        os.makedirs(test_sameset_data_dir+"positive")
        os.makedirs(test_sameset_data_dir+"null")
    if not os.path.exists(test_crossset_data_dir):
        os.makedirs(test_crossset_data_dir)
        os.makedirs(test_crossset_data_dir+"negative")
        os.makedirs(test_crossset_data_dir+"positive")
        os.makedirs(test_crossset_data_dir+"null")

    
    Name_shuffled_trainset = np.array(Name_trainset)
    Arousal_mean_trainset_shuffled = np.array(Arousal_mean_trainset)

    perm = np.random.permutation(len(Name_trainset))
    np.take(Name_trainset,perm,axis=0,out=Name_shuffled_trainset)
    np.take(Arousal_mean_trainset,perm,axis=0,out=Arousal_mean_trainset_shuffled)
    
    
    # no need to shuffle test set
    # Name_shuffled_test = np.array(Name_test)
    # Valence_mean_testset_shuffled = np.array(Valence_mean_testset)
    # perm = np.random.permutation(len(Name_test))
    # np.take(Name_test,perm,axis=0,out=Name_shuffled_test)
    # np.take(Valence_mean_testset,perm,axis=0,out=Valence_mean_testset_shuffled)
    

    Name_trainset_trainsubset, Name_trainset_testsubset = Name_shuffled_trainset[:NTrain],Name_shuffled_trainset[NTrain:]
    Name_trainset_validationsubset = Name_trainset_trainsubset[NValidation:] #take out the image use for validation from train
    Name_trainset_trainsubset = Name_trainset_trainsubset[:NValidation]
    Name_testset = Name_testset
    
    
    Label_trainset_trainsubset, Label_trainset_testsubset = Arousal_mean_trainset_shuffled[:NTrain],Arousal_mean_trainset_shuffled[NTrain:]
    Label_trainset_validationsubset = Label_trainset_trainsubset[NValidation:]
    Label_trainset_trainsubset = Label_trainset_trainsubset[:NValidation]
    Label_testset = Arousal_mean_testset
    
    for i in range(0,len(Name_trainset_trainsubset)):
        if dir_train == dir_gaped:
           subdir_trainsubset = Name_trainset_trainsubset[i][0]
        else :
           subdir_trainsubset = 'F'
        if (float(Label_trainset_trainsubset[i])<0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_trainsubset[i]+".jpeg", train_data_dir+"negative")
        elif (float(Label_trainset_trainsubset[i])>0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_trainsubset[i]+".jpeg", train_data_dir+"positive")
        elif (float(Label_trainset_trainsubset[i])==0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_trainsubset[i]+".jpeg", train_data_dir+"null")
            
    for i in range(0,len(Name_trainset_validationsubset)):
        if dir_train == dir_gaped:
           subdir_trainsubset = Name_trainset_validationsubset[i][0]
        else :
           subdir_trainsubset = 'F'
        if (float(Label_trainset_validationsubset[i])<0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_validationsubset[i]+".jpeg", validation_data_dir+"negative")
        elif (float(Label_trainset_validationsubset[i])>0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_validationsubset[i]+".jpeg", validation_data_dir+"positive")
        elif (float(Label_trainset_validationsubset[i])==0):
            copy(dir_train + subdir_trainsubset+"/"+Name_trainset_validationsubset[i]+".jpeg", validation_data_dir+"null")
    
    for i in range(0,len(Name_trainset_testsubset)):
        if dir_train == dir_gaped:
           subdir_testsubset = Name_trainset_testsubset[i][0]
        else :
           subdir_testsubset = 'F'
        if (float(Label_trainset_testsubset[i])<0):
            copy(dir_train + subdir_testsubset+"/"+Name_trainset_testsubset[i]+".jpeg", test_sameset_data_dir+"negative")
        elif (float(Label_trainset_testsubset[i])>0):
            copy(dir_train + subdir_testsubset+"/"+Name_trainset_testsubset[i]+".jpeg", test_sameset_data_dir+"positive")
        elif (float(Label_trainset_testsubset[i])==0):
            copy(dir_train + subdir_testsubset+"/"+Name_trainset_testsubset[i]+".jpeg", test_sameset_data_dir+"null")
            
    for i in range(0,len(Name_testset)):
        if dir_test == dir_gaped:
           subdir_testcrossset = Name_testset[i][0]
        else :
           subdir_testcrossset = 'F'
        if (float(Label_testset[i])<0):
            copy(dir_test + subdir_testcrossset+"/"+Name_testset[i]+".jpeg", test_crossset_data_dir+"negative")
        elif (float(Label_testset[i])>0):
            copy(dir_test + subdir_testcrossset+"/"+Name_testset[i]+".jpeg", test_crossset_data_dir+"positive")
        elif (float(Label_testset[i])==0):
            copy(dir_test + subdir_testcrossset+"/"+Name_testset[i]+".jpeg", test_crossset_data_dir+"null")

    

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
    print('Train/Test data loaded')
# Initiate the train and test generators with data Augumentation 
    train_datagen = ImageDataGenerator(
    rescale = 1./255)
#    horizontal_flip = True,
#    zoom_range = 0.3,
#    width_shift_range = 0.3,
#    height_shift_range=0.3,)
    
    validation_datagen = ImageDataGenerator(
    rescale = 1./255)
#    horizontal_flip = True,
#    zoom_range = 0.3,
#    width_shift_range = 0.3,
#    height_shift_range=0.3,)
    
    test_sameset_datagen = ImageDataGenerator(
    rescale = 1./255)
#    horizontal_flip = True,
#    zoom_range = 0.3,
#    width_shift_range = 0.3,)
    
    test_crossset_datagen = ImageDataGenerator(
    rescale = 1./255)
#    horizontal_flip = True,
    
    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")
    
    validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical")
    
    test_sameset_generator = test_sameset_datagen.flow_from_directory(
    test_sameset_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")
    
    test_crossset_generator = test_crossset_datagen.flow_from_directory(
    test_crossset_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")
    
    train_features = np.zeros(shape=(NTrain, 11, 14, 2048))
    train_labels = np.zeros(shape=(NTrain,3))
    

    i = 0
    for inputs_batch, labels_batch in train_generator:
        if i * batch_size < NTrain:
            features_batch = model.predict(inputs_batch)
            train_features[i * batch_size : (i + 1) * batch_size] = features_batch
            train_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
        else:        
            break         
    train_features = np.reshape(train_features, (NTrain, 11 * 14 * 2048))
    
    validation_features = np.zeros(shape=(NValidation, 11, 14, 2048))
    validation_labels = np.zeros(shape=(NValidation,3))
    
    i = 0
    for inputs_batch, labels_batch in validation_generator:
        if i * batch_size < NValidation:
            features_batch = model.predict(inputs_batch)
            validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
            validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
        else:        
            break         
    validation_features = np.reshape(validation_features, (NValidation, 11 * 14 * 2048))
    # no need to shuffle test set
    # Name_shuffled_test = np.array(Name_test)
    # Valence_mean_testset_shuffled = np.array(Valence_mean_testset)
    # perm = np.random.permutation(len(Name_test))
    # np.take(Name_test,perm,axis=0,out=Name_shuffled_test)
    # np.take(Valence_mean_testset,perm,axis=0,out=Valence_mean_testset_shuffled)
    

    Name_trainset_trainsubset, Name_trainset_testsubset = Name_shuffled_trainset[:NTrain],Name_shuffled_trainset[NTrain:]    
    Label_trainset_trainsubset, Label_trainset_testsubset = Arousal_mean_trainset_shuffled[:NTrain],Arousal_mean_trainset_shuffled[NTrain:]
    Label_testset = Arousal_mean_testset
   

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
    # one-hot encode the labels
#    y_train = np_utils.to_categorical(y_train, 1024)
#    y_test = np_utils.to_categorical(y_test, 1024)
    
    from keras.callbacks import ModelCheckpoint   
    from keras.layers import Dense, Dropout, Conv2D, GlobalAveragePooling2D,Flatten
    from keras.models import Sequential

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
    model.add(Conv2D(filters=512, kernel_size=2, input_shape=train_features.shape[1:]))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation ='softmax'))
    model.summary()


    #Adding custom Layers 
# =============================================================================
#     x = model.output
#     x = Conv2D(filters=512, kernel_size=2)(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(128, activation="relu")(x)
#     x = Dropout(0.5)(x)
#     predictions = Dense(3, activation="softmax")(x)
#     
#     # creating the final model 
#     model = model(input = model.input, output = predictions)
# =============================================================================

    model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.001),
                  metrics= ['accuracy'])
#    model.compile(loss='mean_squared_error',optimizer='adam', metrics= ['mean_squared_error'])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
    ModelCheckpoint(filepath='model.best.hdf5',verbose=2, save_best_only=True),]
    
    # Train the model 
#    model.fit_generator(train_generator,samples_per_epoch = 20,epochs = 300,validation_data = validation_generator,
#    nb_val_samples = NValidation, callbacks = callbacks)
    model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels),
                    verbose = 2)
    # use tensorboard for displaying the real time evolition of error

    # load the weights that yielded the best validation accuracy
    model.load_weights('model.best.hdf5')
    
    # evaluate test accuracy
    score_sameset = model.evaluate_generator(test_sameset_generator, verbose=0)
    accuracy_sameset = score_sameset[1]*100

    score_crossset = model.evaluate_generator(test_crossset_generator, verbose=0)
    accuracy_crossset = score_crossset[1]*100
    
    # print test accuracy
    print('Test accuracy (same set): %f' % accuracy_sameset)
    print('Test accuracy (cross set): %f' % accuracy_crossset)

    TestResultList.append(accuracy_sameset)
    TestResultList.append(accuracy_crossset)

print(TestResultList)