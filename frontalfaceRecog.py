import cv2
import numpy as np
import os
from sklearn.neighbors import  KNeighborsClassifier

cap = cv2.VideoCapture(0)
frontal_face = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

dataset_path = './data/'

face_data = []
labels = []

names = {}
classification_id = 0

for fx in os.listdir(dataset_path) :
    if not(fx.endswith('.npy'))  :
        continue

    names[classification_id] = fx[:-4]
    print('Loaded ' + fx)
    data_item = np.load(dataset_path + fx)
    face_data.append(data_item)

    target = classification_id * np.ones(data_item.shape[0])
    classification_id += 1
    labels.append(target)

x_train = np.concatenate(face_data, axis = 0)
y_train = np.concatenate(labels, axis = 0)

clf = KNeighborsClassifier(5)
clf.fit(x_train, y_train)

while True :
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces = frontal_face.detectMultiScale(frame, 1.1, 5)
    offset = 10
    
    for (x, y, w, h) in faces :
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 2)
        
        face_section = frame[y - offset : y + h + offset, x - offset : x + w + offset]
        try : 
            face_section = cv2.resize(face_section, (100, 100))
        except Exception :
            break

        predict_label = clf.predict([face_section.flatten()])
        predict_name = names[int(predict_label[0])]

        cv2.putText(frame, predict_name, (x ,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video_Classified', frame)

    key = cv2.waitKey(1)
    if key == ord('q') :
        break

    

cap.release()
cv2.destroyAllWindows()


    