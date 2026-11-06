import cv2
import time

# Import database functions
from database import (
    create_tables,
    add_person,
    save_embedding,
    get_all_embeddings
)

# Import face recognition functions
from face_engine import (
    initialize_face_engine,
    detect_faces,
    generate_embedding,
    recognize_face
)


# ==========================================
# 1. REGISTER A NEW PERSON
# ==========================================

def register_new_person(cap, face_app):

    name = input("\nEnter person's name: ").strip()

    if not name:
        print("Error: Name cannot be empty!")
        return False

    collected_embeddings = []

    total_samples = 15

    print(f"\nRegistering {name}...")
    print("Look at the camera and slightly move your face.")
    print("Press Q to cancel registration.\n")

    while len(collected_embeddings) < total_samples:

        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture webcam frame.")
            return False

        faces = detect_faces(face_app, frame)

        # Accept only one detected face
        if len(faces) == 1:

            face = faces[0]

            # Check face detection confidence
            if face.det_score >= 0.7:

                embedding = generate_embedding(face)

                collected_embeddings.append(embedding)

        # Display registration progress
        progress = len(collected_embeddings)

        cv2.putText(
            frame,
            f"Registration: {progress}/{total_samples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Registration cancelled.")
            return False

        time.sleep(0.1)

    # Create person record in SQLite
    person_id = add_person(name)

    if person_id is None:
        return False

    # Save collected embeddings
    for embedding in collected_embeddings:

        save_embedding(
            person_id,
            embedding
        )

    print("\n================================")
    print(f"Registration successful: {name}")
    print(f"Total face samples saved: {total_samples}")
    print("================================\n")

    return True


# ==========================================
# 2. MAIN APPLICATION
# ==========================================

def main():

    print("\n================================")
    print("   FACE RECOGNITION SYSTEM")
    print("================================\n")

    # Create SQLite tables
    create_tables()

    # Initialize InsightFace model
    face_app = initialize_face_engine()

    # Load registered face embeddings
    database_embeddings = get_all_embeddings()

    print(
        f"Loaded {len(database_embeddings)} face samples "
        "from database."
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Error: Could not open webcam!")

        return

    print("\nWebcam started successfully!")

    print("\nControls:")
    print("R = Register unknown person")
    print("Q = Quit application\n")

    # ======================================
    # 3. REAL-TIME FACE RECOGNITION LOOP
    # ======================================

    while True:

        # Capture webcam frame
        ret, frame = cap.read()

        if not ret:

            print("Error: Unable to read webcam frame.")
            break

        # Detect faces
        faces = detect_faces(
            face_app,
            frame
        )

        unknown_face_present = False

        # Process every detected face
        for face in faces:

            # Get bounding box coordinates
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Generate live face embedding
            live_embedding = generate_embedding(face)

            # Compare with registered faces
            result = recognize_face(
                live_embedding,
                database_embeddings
            )

            if result is not None:

                # Known person
                name = result["name"]

                similarity = result["similarity"]

                label = f"{name} ({similarity:.2f})"

                color = (0, 255, 0)

            else:

                # Unknown person
                label = "Unknown - Press R"

                color = (0, 165, 255)

                unknown_face_present = True

            # Draw rectangle around detected face
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display recognition result
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Display detected face count
        cv2.putText(
            frame,
            f"Faces Detected: {len(faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display webcam window
        cv2.imshow(
            "Face Recognition System",
            frame
        )

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Quit application
        if key == ord("q"):

            print("\nClosing application...")
            break

        # Register unknown person
        elif key == ord("r"):

            if len(faces) == 1 and unknown_face_present:

                success = register_new_person(
                    cap,
                    face_app
                )

                if success:

                    # Reload embeddings from SQLite
                    database_embeddings = get_all_embeddings()

                    print("Face database refreshed!")

            elif len(faces) == 0:

                print("No face detected. Please face the camera.")

            else:

                print(
                    "Registration requires exactly one "
                    "unknown face in front of the camera."
                )

    # ======================================
    # 4. RELEASE RESOURCES
    # ======================================

    cap.release()

    cv2.destroyAllWindows()

    print("\nFace Recognition System stopped.")


# ==========================================
# 5. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()

