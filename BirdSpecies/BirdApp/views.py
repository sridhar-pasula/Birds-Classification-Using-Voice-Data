from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os
import librosa
from sklearn.preprocessing import StandardScaler #class to normalized bird fetaures
import cv2
from keras.callbacks import ModelCheckpoint
import keras
from keras import Model, layers

global classifier
labels = ['Buff-throated Woodcreeper', 'Lesser Woodcreeper', 'Olivaceous Woodcreeper', 'Yellow-olive Flatbill']
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
#dataset preprocessing such as normalization, dataset shuffling and train and test split
scaler = StandardScaler()
X = scaler.fit_transform(X)#normalizing or scaling dataset values
X = np.reshape(X, (X.shape[0], 128, 87, 1))
'''
#now train Alexnet with CNN 5 layers and added extra 3 layers with number of neurons as 9,9 and 7,7 and 6,6
classifier = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X.shape[1],X.shape[2],X.shape[3])),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(9,9), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(7,7), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(6,6), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(1,1), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(4, activation='softmax')
])
classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])#compiling the model
#training and loading the model
if os.path.exists("model/model_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/model_weights.hdf5', verbose = 1, save_best_only = True)
    hist = classifier.fit(X_train, y_train, batch_size = 8, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    classifier = load_model("model/model_weights.hdf5")
'''
def UploadVoiceAction(request):
    if request.method == 'POST':
        global classifier, scaler, labels
        content = request.FILES['t1'].read()
        if os.path.exists('BirdApp/static/test.mp3'):
            os.remove('BirdApp/static/test.mp3')
        with open('BirdApp/static/test.mp3', "wb") as file:
            file.write(content)
        file.close()
        audio, sr = librosa.load(path='BirdApp/static/test.mp3', sr=None)#now read audio wav file
        audio = librosa.effects.time_stretch(y=audio, rate=len(audio)/sr) #extract sample rate from the audio
        # Calculate features and get the label from the filename
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)#now extract sepctogram features
        mels_db = librosa.power_to_db(S=mels, ref=1.0)
        test = []
        test.append(mels_db.ravel()) #now convert multi dimension fetaures to 2 dimension fetaures for normalization
        test = np.asarray(test)
        test = scaler.transform(test) #normalizing test audio fetaurfes
        test = np.reshape(test, (test.shape[0], 128, 87, 1))
        classifier = load_model("model/model_weights.hdf5")
        predict = classifier.predict(test) #perform prediction on test bird audio file
        predict = np.argmax(predict, axis=1)
        dataset = "Given Bird Voice Identified as : "+labels[predict[0]]
        title = "Given Bird Voice Identified as : "+labels[predict[0]] #plot image of predicted bird audio file
        img = cv2.imread("images/"+str(predict[0])+".jpg")
        img = cv2.resize(img, (700,400))
        cv2.putText(img, title, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 0, 0), 2)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        context= {'data':str(dataset)}
        return render(request, 'UploadVoice.html', context)

def UploadVoice(request):
    if request.method == 'GET':
       return render(request, 'UploadVoice.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global uname, email_id
        option = 0
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'UserLogin.html', context)        

