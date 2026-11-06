
import cv2
from insightface.app import FaceAnalysis

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(640, 640))

print("Model loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        print("Embedding shape:", face.normed_embedding.shape)

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
