import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (255, 0, 0), 4)

        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            radius = int((round(ew + eh) * 0.25))
            cv2.ellipse(roi_color, (ex + ew // 2, ey + eh // 2), (radius, radius), 0, 0, 360, (0, 255, 0), 5)


    if len(faces) > 0 and len(eyes) > 0:
        cv2.putText(frame, 'POGI SIGE NA  ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('EXAM AND QUIZ', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
