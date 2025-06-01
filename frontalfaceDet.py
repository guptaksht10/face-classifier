import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

name = input('Enter the name of the person: ')

face_data = []
counter = 0

while True :
    ret, frame = cap.read()

    if ret == False :
        continue

    faces = face_cascade.detectMultiScale(frame, 1.1, 5)
    for (x, y, w, h) in faces :
        cv2.rectangle(frame, (x, y), (x + w, y + h), [0, 0, 255], 2)

    show_frame = frame[:, ::-1, :]

    cv2.imshow('Face_Capture', show_frame)
    key = cv2.waitKey(1)

    if key == ord('q') :
        break
        
    if len(faces) == 0:
        continue

    face = sorted(faces, key = lambda x: x[2] * x[3])[-1]
    x, y, w, h = face

    offset = 10
    face_image = frame[y - offset : y + h + offset, x - offset : x + w + offset] # cropped_image
    
    try :
        face_image = cv2.resize(face_image, (100, 100))
    except Exception :
        continue
        
    show_face_image = face_image[:, ::-1, :]
    cv2.imshow("Cropped Image", show_face_image)

    if counter % 10 == 0 :
        face_data.append(face_image)

    counter += 1

face_data = np.array(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)

np.save('./data/'+name+'.npy',face_data)
print('Data Saved Successfully !!')
    
cap.release()
cv2.destroyAllWindows()
