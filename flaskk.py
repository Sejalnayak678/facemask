from flask import Flask, render_template, jsonify
import test
import cv2
import numpy as np
import requests
from flask import request
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import imutils
import tkinter
from tkinter import messagebox
import smtplib
root=tkinter.Tk()
import pygame
root.withdraw()
import matplotlib.pyplot as plt
from keras.models import load_model
app = Flask(__name__)



@app.route('/', methods=['POST', 'GET'])
def myfacemask():
    if request.method == "GET":
    
        model=load_model("model2-009.model")

        labels_dict={0:'without mask',1:'mask'}
        color_dict={0:(0,0,255),1:(0,255,0)}

        size = 4
        webcam = cv2.VideoCapture(0)


        classifier = cv2.CascadeClassifier('/Users/sejalnayak/Desktop/pi/haarcascade_frontalface_default.xml')

        SUBJECT="Subject"
        TEXT="One visitor violated Face Mask policy."


        pygame.init()
        pygame.mixer.init()
        sounda= pygame.mixer.Sound("/Users/sejalnayak/Desktop/beep/beep-01a.wav")


        while True:
            (rval, im) = webcam.read()
            im=cv2.flip(im,1,1) #Flip to act as a mirror

            # Resize the image to speed up detection
            mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

            # detect MultiScale / faces 
            faces = classifier.detectMultiScale(mini)

            # Draw rectangles around each face
            for f in faces:
                (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
                #Save just the rectangle faces in SubRecFaces
                face_img = im[y:y+h, x:x+w]
                resized=cv2.resize(face_img,(150,150))
                normalized=resized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)
                #print(result)
                
                label=np.argmax(result,axis=1)[0]
            
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
                if(label==0):
                    sounda.play()
                    ##messagebox.showwarning("Warning","Access Denied,Please wear a mask")
                    ##message = 'Subject: {}\n\n{}'.format(SUBJECT,TEXT)
                    ##mail=smtplib.SMTP('smtp.gmail.com',587)
                    ##mail.ehlo()
                    ##mail.starttls()
                    ##mail.login('infraking.sn@gmail.com','satish7691')
                    ##mail.sendmail('infraking.sn@gmail.com','infraking.sn@gmail.com',message)
                    ##mail.close
                    
                    
                ##elif(label!=0):
                    ##messagebox.showwarning("entry allowed")
                ##else:
                    ##pass
                    
                
            # Show the image
            cv2.imshow('LIVE',   im)
            key = cv2.waitKey(10)
            # if Esc key is press then break out of the loop 
            if key == 27: #The Esc key
                break
        # Stop video
        webcam.release()

        # Close all started windows
        cv2.destroyAllWindows()

    return render_template("hy.html")


    if __name__== "main":
        app.run(debug=True)

