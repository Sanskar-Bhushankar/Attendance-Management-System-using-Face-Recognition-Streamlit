import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to training images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print("Training Images:", myList)

# Load training images and extract class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Class Names:", classNames)


# Function to encode faces from images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Function to mark attendance
def markAttendance(name):
    # Create the file if it doesn't exist
    if not os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')  # Write header row

    # Read the file and update attendance
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Capture video from webcam
cap = cv2.VideoCapture(0)

detected_name = None

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce size for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and encode them
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            detected_name = classNames[matchIndex].upper()

            # Draw rectangle around the face and display name
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, detected_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Stop after detecting and displaying the name
            break

    # Display the webcam feed
    cv2.imshow('Webcam', img)

    if detected_name or cv2.waitKey(1) & 0xFF == ord('q'):  # Break loop if a name is detected or 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

# Mark attendance after closing the camera
if detected_name:
    markAttendance(detected_name)
    print(f"Attendance marked for: {detected_name}")
else:
    print("No face detected.")
