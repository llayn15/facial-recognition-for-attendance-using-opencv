import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Absen'
images = []
classNames = []
myList = os.listdir(path)

# label untuk tiap berkas
label_mapping = {
    '1.png': {'name': 'marco', 'nim': '14S21025'},
    '2.png': {'name': 'rizki', 'nim': '14S21040'},
    '3.png': {'name': 'perez', 'nim': '14S21038'},
}

for cl in myList:
    if cl.lower().endswith(('.jpg', '.jpeg', '.png')):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)

        if cl in label_mapping:
            data = label_mapping[cl]
            classNames.append({'name': data['name'], 'nim': data['nim']})

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, nim):
    with open('Absensi.txt', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')

        # Mengecek apakah nama sudah tercatat sebelumnya
        with open('Absensi.txt', 'r') as f_read:
            lines = f_read.readlines()
            for line in lines:
                if name in line:
                    return  # Jika nama sudah tercatat, keluar dari fungsi

        f.write(f'{name}, {nim}, {dtString}\n')

# Menambahkan variabel untuk menyimpan nama dan nim wajah terakhir yang terdeteksi
last_detected_face = None

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
camera_active = True  # Menambahkan variabel untuk memantau apakah kamera aktif
capture_mode = False  # Menambahkan variabel untuk menentukan apakah mode capture aktif

def on_mouse_click(event, x, y, flags, param):
    global capture_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_mode = not capture_mode

# Mendaftarkan fungsi mouse event
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', on_mouse_click)

while camera_active:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    success, img = cap.read()

    if capture_mode:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Menyimpan indeks wajah yang sudah terdeteksi
        detected_face_indices = []

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            if matches[matchIndex]:
                data = label_mapping[myList[matchIndex]]
                name = data['name'].upper()
                nim = data['nim']

                # Mengecek apakah wajah yang terdeteksi adalah berbeda dengan wajah terakhir
                if last_detected_face != name and name not in detected_face_indices:
                    markAttendance(name, nim)
                    last_detected_face = name
                    detected_face_indices.append(name)

                # Draw green rectangle for recognized faces
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            else:
                # Draw red rectangle for unrecognized faces
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, 'Unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

cap.release()
cv2.destroyAllWindows()