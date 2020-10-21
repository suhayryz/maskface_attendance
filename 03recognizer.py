# -*- coding: utf-8 -*-
"""
@author: suhairisuhaimin
"""

import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import time
import datetime

# images properties
def plt_show(image, title=""):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.title(title)
    plt.imshow(image, cmap="Greys_r")
    plt.show()

# face detection    
class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (75, 75)
        biggest_only = True
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT |                     cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else                     cv2.CASCADE_SCALE_IMAGE
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=flags)
        return faces_coord

# video camera
class VideoCamera(object):
    def __init__(self, index=1):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print (self.video.isOpened())

    def __del__(self):
        self.video.release()
    
    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

# crop images
def cut_faces(image, faces_coord):
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

# normalize images
def normalize_intensity(images):
    images_norm = []
    for image in images:
        is_color = len(image.shape) == 3 
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images_norm.append(cv2.equalizeHist(image))
    return images_norm

# resize images
def resize(images, size=(100, 100)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

# normalize faces
def normalize_faces(frame, faces_coord):
    faces = cut_faces(frame, faces_coord)
    faces = normalize_intensity(faces)
    faces = resize(faces)
    return faces

# rectangle line
def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:
        w_rm = int(0.2 * w / 2) 
        cv2.rectangle(image, (x + w_rm, y), (x + w - w_rm, y + h), 
                              (102, 255, 0), 1)
# acquire images from dataset
def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    members = [person for person in os.listdir("members/")]
    for i, person in enumerate(members):   # loop over
        labels_dic[i] = person
        for image in os.listdir("members/" + person):
            images.append(cv2.imread("members/" + person + '/' + image, 
                                     0))
            labels.append(i)
    return (images, np.array(labels), labels_dic)

images, labels, labels_dic = collect_dataset()


# train image (algorithm sets)
rec_eig = cv2.face.EigenFaceRecognizer_create()
rec_eig.train(images, labels)

rec_fisher = cv2.face.FisherFaceRecognizer_create()
rec_fisher.train(images, labels)

rec_lbph = cv2.face.LBPHFaceRecognizer_create()
rec_lbph.train(images, labels)

print ("Models Trained Succesfully")


# cascade face and mask
detector = FaceDetector("xml/frontal_face.xml")
detector_mask = cv2.CascadeClassifier("xml/mask_cascade.xml")
# 0 usb webcam additional
# 1 back cam acer
# 2 front cam acer
webcam0 = VideoCamera(0)
webcam1 = VideoCamera(1)


ts = time.time()      
date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')


# mask detection and face recognition (in)
while True:    
    frame0 = webcam0.get_frame()
    mask = detector_mask.detectMultiScale(frame0, 
                                 scaleFactor=1.2, 
                                 minNeighbors=5, 
                                 minSize=(100, 100),
                                 maxSize=(150, 150),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    for(x1,y1,x2,y2) in mask:
        cv2.rectangle(frame0,(x1,y1),(x1+x2,y1+y2),(0,255,0),2)
        cv2.putText(frame0, 'Using Mask',(x1, y1+y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    faces_coord = detector.detect(frame0, False) # detect more than one face
    col_names =  ['Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)
    if len(faces_coord):
        faces = normalize_faces(frame0, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.StandardCollector_create()
            rec_lbph.predict_collect(face, collector) # chosen algorithm
            conf = collector.getMinDist()
            pred = collector.getMinLabel()
            threshold = 76 # eigen 2600, fisher 1600, lbph 140 [mean 3536,1285,68] [lbph lowest 76]
            print ("Prediction Entry: " + labels_dic[pred].capitalize() + "\nConfidence Entry: " + str(round(conf)))
            
            if conf < threshold: # apply threshold
                cv2.putText(frame0, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (102, 255, 0), 1)
                attendance.loc[len(attendance)] = [labels_dic[pred],date,timeStamp]
                Hour,Minute,Second=timeStamp.split(":")
                fileName="attendancein\Attendance_"+labels_dic[pred]+"-"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"  # write to output file (in)
                attendance.to_csv(fileName,index=False)
  
            else:
                cv2.putText(frame0, "Unknown",
                    (faces_coord[i][0], faces_coord[i][1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (66, 55, 245), 1)
        draw_rectangle(frame0, faces_coord) # rectangle around face
    cv2.putText(frame0, "ESC to exit", (5, frame0.shape[0] - 5),
    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Entry Cam", frame0) # live feed in external
    if cv2.waitKey(33) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
           
# mask detection and face recognition (out)   
 
    frame1 = webcam1.get_frame()
    mask = detector_mask.detectMultiScale(frame1, 
                                 scaleFactor=1.2, 
                                 minNeighbors=5, 
                                 minSize=(100, 100),
                                 maxSize=(150, 150),
                                 flags=cv2.CASCADE_SCALE_IMAGE)
    for(x1,y1,x2,y2) in mask:
        cv2.rectangle(frame1,(x1,y1),(x1+x2,y1+y2),(0,255,0),2)
        cv2.putText(frame1, 'Using Mask',(x1, y1+y2 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    
    faces_coord = detector.detect(frame1, False) # detect more than one face
    col_names =  ['Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)
    if len(faces_coord):
        faces = normalize_faces(frame1, faces_coord) # norm pipeline
        for i, face in enumerate(faces): # for each detected face
            collector = cv2.face.StandardCollector_create()
            rec_lbph.predict_collect(face, collector)  # chosen algorithm
            conf = collector.getMinDist()
            pred = collector.getMinLabel()
            threshold = 76     # eigen 2600, fisher 1600, lbph 140 [mean 3536,1285,68] [lbph lowest 76]
            print ("Prediction Exit: " + labels_dic[pred].capitalize() + "\nConfidence Exit: " + str(round(conf)))
            
            if conf < threshold: # apply threshold
                cv2.putText(frame1, labels_dic[pred].capitalize(),
                            (faces_coord[i][0], faces_coord[i][1] - 20),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (102, 255, 0), 1)
                attendance.loc[len(attendance)] = [labels_dic[pred],date,timeStamp]
                Hour,Minute,Second=timeStamp.split(":")
                fileName="attendanceout\Attendance_"+labels_dic[pred]+"-"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"    # write to output file (out)
                attendance.to_csv(fileName,index=False)
  
            else:
                cv2.putText(frame1, "Unknown",
                    (faces_coord[i][0], faces_coord[i][1] - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (66, 55, 245), 1)
        draw_rectangle(frame1, faces_coord) # rectangle around face
    cv2.putText(frame1, "ESC to exit", (5, frame1.shape[0] - 5),
    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Exit Cam", frame1) # live feed in external
    if cv2.waitKey(33) & 0xFF == 27:
        cv2.destroyAllWindows()
        break
      
del webcam0
del webcam1
