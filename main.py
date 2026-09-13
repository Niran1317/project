import os
import cv2
import numpy as np

MODEL_DIR = "models"

FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
            "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def check_models():
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL,
                GENDER_PROTO, GENDER_MODEL]
    missing = [x for x in required if not os.path.exists(x)]
    if missing:
        print("Missing model files:")
        for item in missing:
            print(" -", item)
        print("\nSee models/README.md for the required files.")
        return False
    return True

def detect_and_predict(frame, face_net, age_net, gender_net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), [104, 117, 123], swapRB=False
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.60:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        gender_net.setInput(face_blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        age_net.setInput(face_blob)
        age = AGE_LIST[age_net.forward()[0].argmax()]

        label = f"{gender}, {age}"
        results.append((x1, y1, x2, y2, label))

    return results

def run_webcam():
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press Q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        results = detect_and_predict(frame, face_net, age_net, gender_net)
        for x1, y1, x2, y2, label in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Age & Gender Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if check_models():
        run_webcam()
