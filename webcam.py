"""
Step 2: Use your webcam to detect your emotion in real time.

Run:  python webcam.py
Press 'q' to quit.
"""
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 48

# Load the trained CNN and the emotion names.
model = tf.keras.models.load_model("emotion_model.keras")
with open("labels.txt") as f:
    labels = f.read().splitlines()

# OpenCV ships with a ready-made face detector (a "Haar cascade").
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)  # 0 = your default camera
if not camera.isOpened():
    raise SystemExit("Could not open the camera. Check your camera permissions.")

while True:
    ok, frame = camera.read()
    if not ok:
        break

    # The model was trained on grayscale images, so convert the frame.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find every face in the frame. Each face is a box: (x, y, width, height).
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Cut out the face and shrink it to 48x48, like the training images.
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")

        # Ask the CNN for a probability for each emotion.
        probs = model(face, training=False).numpy()[0]
        emotion = labels[np.argmax(probs)]
        confidence = np.max(probs)

        # Draw a box and the predicted emotion on the video.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.0%})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Emotion Detector (press q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
