# -*- coding: utf-8 -*-
"""
@author: suhairisuhaimin
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


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
                              (200, 200, 0), 4)

# acquire images from dataset
def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    members = [person for person in os.listdir("members/")]
    for i, person in enumerate(members):
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


# face recognition
webcam = VideoCamera(1)
frame = webcam.get_frame()
detector = FaceDetector("xml/frontal_face.xml")
frame = webcam.get_frame()
faces_coord = detector.detect(frame)
faces = normalize_faces(frame, faces_coord)
face = faces[0]
plt_show(face) 


collector = cv2.face.StandardCollector_create()

rec_eig.predict_collect(face, collector)
conf = collector.getMinDist()
pred = collector.getMinLabel()

print ("Eigen Faces -> Prediction: " + labels_dic[pred] +"    Confidence: " + str(round(conf)))

rec_fisher.predict_collect(face, collector)
conf = collector.getMinDist()
pred = collector.getMinLabel()

print ("Fisher Faces -> Prediction: " +labels_dic[pred] + "    Confidence: " + str(round(conf)))

rec_lbph.predict_collect(face, collector)
conf = collector.getMinDist()
pred = collector.getMinLabel()

print ("LBPH Faces -> Prediction: " +labels_dic[pred] + "    Confidence: " + str(round(conf)))

