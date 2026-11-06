import numpy as np

from insightface.app import FaceAnalysis

#initial similarity thresold
SIMILARITY_THRESOLD = 0.45

def initialize_face_engine():
    """Loads the insightFace model."""
    
    print("Loading face recognition model....")
    
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=["CPUExecutionProvider"]
    )
    
    face_app.prepare(
        ctx_id=-1,
        det_size=(640,640)
    )
    
    print("Face recognition mofdel loaded!")
    
    return face_app

def detect_faces(face_app,image):
    """ Detects all faces present in an image."""
    
    faces = face_app.get(image)
    
    return faces

def generate_embedding(face):
    """ Exctracts the normlized face embedding."""
    
    embedding = face.normed_embedding
    
    return np.asarray(
        embedding,
        dtype=np.float32
    )
    
def compare_embeddings(embedding1,embedding2):
    """Calculates consine similarity between two embeddings."""
    
    embedding1 = np.asarray(embedding1,dtype=np.float32)
    
    embedding2 = np.asarray(embedding2,dtype=np.float32)
    
    norm1 = np.linalg.norm(embedding1)
    
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return -1.0

        similarity = np.dot(embedding1,embedding2)/(norm1 * norm2)
        
        return float(similarity)
    
    
def recognize_face(live_embedding,database_embeddings):
    """ com[pares a live face wiyh all saved embeddings.
    Returns the best matching person if the  similarity exceeds the configures thresold. """
    
    best_match = None
    highest_similarity = -1.0
    
    for record in database_embeddings:
        
        saved_embedding=record["embedding"]
        
        similarity = compare_embeddings(;ive_embeedings,saved_embeddings)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = record
            
        if (best_match is Not None and highest_similarity >= SIMILARITY_THRESOLD):
            return({
                "id" : best_match["id"],
                "name" : best_match["name"],
                "similarity" : highest_similarity
            }) 
            
    return None
    
       
    
    