import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_alt.xml'
datasets = 'datasets'
print('Training...')

(images, labels, names, id) = ([], [], {}, 0)

# Prepare the dataset
for subdir, dirs, _ in os.walk(datasets):
    for dir_name in dirs:
        names[id] = dir_name
        subjectpath = os.path.join(datasets, dir_name)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
            labels.append(label)
        id += 1

(width, height) = (130, 100)

(images, labels) = [np.array(lis) for lis in [images, labels]]

# Initialize face recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    _, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1] < 800:
            cv2.putText(im, f'{names[prediction[0]]} - {prediction[1]:.0f}',
                        (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'unknown', (x - 10, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown person detected")
                cv2.imwrite("input.jpg", im)
                cnt = 0

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:  # Escape key
        break

webcam.release()
cv2.destroyAllWindows()

