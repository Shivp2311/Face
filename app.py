import cv2
import numpy as np
import os
import streamlit as st

size = 2 # change this to 4 to speed up processing trade off is the accuracy
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'
print("Face Recognition Starting ...")

# Create a list of images,labels,dictionary of corresponding names
(images, labels, names, id) = ([], [], {}, 0)

# Get the folders containing the training data
for (subdirs, dirs, files) in os.walk(image_dir):

    # Loop through each folder named after the subject in the photos
    for subdir in dirs:
        names[id] = subdir
        
        subjectpath = os.path.join(image_dir, subdir)

        # Loop through each photo in the folder
        for filename in os.listdir(subjectpath):

            # Skip non-image formats
            f_name, f_extension = os.path.splitext(filename)
            if(f_extension.lower() not in
                    ['.png','.jpg','.jpeg','.gif','.pgm']):
                print("Skipping "+filename+", wrong file type")
                continue
            path = subjectpath + '/' + filename
            label = id

            # Add to training data
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(im_width, im_height) = (120, 102)

# Create a Numpy array from the two lists above
(images, labels) = [np.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
haar_cascade = cv2.CascadeClassifier(classifier)

def detect_faces(frame):
    # Convert to grayscalel
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to speed up detection (optional, change size above)
    mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

    # Detect faces and loop through each one
    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]

        # Coordinates of face after scaling back by size
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        start =(x, y)
        end =(x + w, y + h)

        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(frame,start , end, (0, 255, 0), 3) # creating a bounding box for detected face
        cv2.rectangle(frame, (start[0],start[1]-20), (start[0]+120,start[1]), (0, 255, 255), -3) # creating rectangle on the upper part of bounding box
        #for i in prediction[1]
        if prediction[1] < 90:  # note: 0 is the perfect match, the higher the value the lower the accuracy
            cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),thickness=2)
            print('%s - %.0f' % (names[prediction[0]],prediction[1]))
       
