import sqlite3
import numpy as np

DATABASE_NAME = 'faces.db'

def get_connection():
    """Creates a connection to the sqlite database.
    if the database does not exist, SQLite creates it."""
    
    return sqlite3.connect(DATABASE_NAME)

def create_tables():
    """Creates the database tables if they do not already exist."""
    
    conn = get_connection()
    cursor = conn.cursor()
    
    #Table containing person information 
    cursor.execute("""CREATE TABLE IF NOT EXISTS persons(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)
        """)
    
    #Table containing face embeddings
    cursor.execute("""CREATE TABLE IF NOT EXISTS face_embeddings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        embedding BLOB NOT NULL,
        
        FOREIGN KEY(person_id) REFERENCES persons(id)
        ON DELETE CASCADE)
        """)
    
    conn.commit()
    conn.close()
    
    print("Database tables created successfully!")
    
    def add_person(name):
        """Creates a new person record.
        Returns the person's database ID."""
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO persons (name) VALUES (?)",(name,))
            
            person_id = cursor.lastrowid
            
            conn.commit()
            
            return person_id
        
        except sqlite3.IntegrityError:
            print("This person already exists!")
            
            return None
        
        finally:
            conn.close()
            
    def save_embedding(person_id,embedding):
        """Converts a Numpy embedding ito binary data and stores it in SQLite."""
        
        conn = get_connection()
        cursor = conn.cursor()
        
        #converts embedding to float32
        
        embedding = np.asarray(
            embedding,
            dtype=np.float32)
        
        #convert Numpy array into binary format
        
        embedding_blob = embedding.tobytes()
        
        cursor.execute("""INSERT INTO face_embeddings (person_id, embedding) VALUES (?,?)""",(person_id, embedding_blob))
        
        conn.commit()
        conn.close()
        
    def get_all_embeddings():
        """Retrieves all registered people and their embeddings.
        Converts binary data back into Numpy arrays."""
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""SELECT persons.id, persons.name, face_embeddings.embedding
        FROM persons JOIN face|_embeddings ON persons.id = face_embeddings.persons_id""")
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        
        for person_id, name, embedding_blob in rows:
            
            #Convert binary data back into Numpy array
            
            embedding = np.frombuffer(embedding_blob,dtype = np.float32).copy()
            
            results.append({
                "id": person_id,
                "name" : name,
                "embedding" : embedding
            })
            
            return results
        
    def delete_person(person_id):
        """Deletes a person and their associatedembeddings.
        """
        conn = get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        
        conn.commit()
        conn.close()