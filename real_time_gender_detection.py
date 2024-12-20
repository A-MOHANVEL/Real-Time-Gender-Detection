import cv2
import tensorflow as tf
import numpy as np

model_path = 'C:/Users/Mohanvel/Downloads/gender_detection_model.h5'
model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        
        resized_face = cv2.resize(face, (150, 150)) / 255.0
        img_array = np.expand_dims(resized_face, axis=0)
        
        prediction = model.predict(img_array)
        
        female_prob = prediction[0][0]  
        male_prob = 1 - female_prob     
        if abs(female_prob - male_prob) < 0.1:
            continue

        gender = "Female" if female_prob < male_prob else "Male"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gender Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cv2.waitKey(1) 