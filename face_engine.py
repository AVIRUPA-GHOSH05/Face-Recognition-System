
import numpy as np

from insightface.app import FaceAnalysis

# Initial similarity threshold
SIMILARITY_THRESHOLD = 0.45


def initialize_face_engine():
    """Loads the InsightFace model."""

    print("Loading face recognition model....")

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    print("Face recognition model loaded!")

    return face_app


def detect_faces(face_app, image):
    """Detects all faces present in an image."""

    faces = face_app.get(image)

    return faces


def generate_embedding(face):
    """Extracts the normalized face embedding."""

    embedding = face.normed_embedding

    return np.asarray(
        embedding,
        dtype=np.float32
    )


def compare_embeddings(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""

    embedding1 = np.asarray(embedding1, dtype=np.float32)
    embedding2 = np.asarray(embedding2, dtype=np.float32)

    if embedding1.shape != embedding2.shape:
        return None

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return None

    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    return float(similarity)


def recognize_face(live_embedding, database_embeddings):
    """
    Compares a live face with all saved embeddings.

    Returns the best matching person if the similarity
    exceeds the configured threshold.
    """

    best_match = None
    highest_similarity = -1.0

    for record in database_embeddings:

        saved_embedding = record["embedding"]

        similarity = compare_embeddings(
            live_embedding,
            saved_embedding
        )

        if similarity is None:
            continue

        if similarity > highest_similarity:

            highest_similarity = similarity
            best_match = record

    # Check the threshold AFTER comparing all saved samples
    if (
        best_match is not None
        and highest_similarity >= SIMILARITY_THRESHOLD
    ):

        return {
            "id": best_match["id"],
            "name": best_match["name"],
            "similarity": highest_similarity
        }

    return None
