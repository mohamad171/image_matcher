import cv2
import numpy as np
import faiss
import sqlite3

conn = sqlite3.connect('features.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS features
             (id INTEGER PRIMARY KEY AUTOINCREMENT, descriptor BLOB)''')
def extract_features(image):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors

def store_features(descriptors):
        for descriptor in descriptors:
                # Convert descriptor to binary
                descriptor_blob = descriptor.tobytes()
                c.execute("INSERT INTO features (descriptor) VALUES (?)", (descriptor_blob,))
        conn.commit()

        # Retrieve all descriptors to create FAISS index
        c.execute("SELECT descriptor FROM features")
        all_descriptors = c.fetchall()
        all_descriptors = np.array([np.frombuffer(d[0], dtype=np.uint8) for d in all_descriptors])

        d = 32  # Dimension of descriptors
        index = faiss.IndexFlatL2(d)
        index.add(all_descriptors.astype('float32'))
        faiss.write_index(index, "faiss_index.bin")
        return index

def load_faiss_index():
    return faiss.read_index("faiss_index.bin")

def search_similar(image, index, threshold=0.7):
    descriptors = extract_features(image)
    distances, indices = index.search(descriptors.astype('float32'), 5)  # Search for top 5 matches
    if np.min(distances) < threshold:
        return indices[0]  # Return the closest match
    else:
        return None


if __name__ == "__main__":
    # Load your image here
    image = cv2.imread('sample_images/1.jpg', 0)

    # Extract and store features
    descriptors = extract_features(image)
    store_features(descriptors)

    # Load FAISS index
    index = load_faiss_index()

    # Search for similar images
    result = search_similar(image, index)
    print("Matching indices:", result)