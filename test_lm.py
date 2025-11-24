import cv2, mediapipe as mp

face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face.process(rgb)

    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0].landmark
        print("TOTAL LANDMARKS =", len(lms))
        break

cap.release()
ß
