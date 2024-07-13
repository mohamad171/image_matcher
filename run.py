import sqlite3
import cv2
import numpy as np
import faiss
import random
# Connect to SQLite database (create if it doesn't exist)
conn = sqlite3.connect('image_features.db')
c = conn.cursor()

# Create table for storing image descriptors
c.execute('''
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor BLOB,
    server_id INTEGER
)
''')
conn.commit()

def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path, 0)  # 0 to read in grayscale
    # Initialize ORB detector
    orb = cv2.ORB_create()
    # Detect and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors

def store_features(descriptors,server_id):
    for descriptor in descriptors:
        # Convert descriptor to binary
        descriptor_blob = descriptor.tobytes()
        c.execute("INSERT INTO features (descriptor,server_id) VALUES (?,?)", (descriptor_blob,server_id))
    conn.commit()

def load_descriptors():
    c.execute("SELECT * FROM features")
    rows = c.fetchall()
    ids = []
    server_ids = []
    all_descriptors = []
    for row in rows:
        ids.append(row[0])
        server_ids.append(row[2])
        descriptor = np.frombuffer(row[1], dtype=np.uint8)
        all_descriptors.append(descriptor)
    all_descriptors = np.array(all_descriptors)
    return server_ids,ids, all_descriptors

def build_faiss_index(descriptors):
    d = descriptors.shape[1]  # Dimension of descriptors
    index = faiss.IndexFlatL2(d)
    index.add(descriptors.astype('float32'))
    return index

def search_similar(image_path, index, ids,server_ids, threshold=1000):
    descriptors = extract_features(image_path)
    distances, indices = index.search(descriptors.astype('float32'), 1)  # Search for the closest match
    if np.min(distances) < threshold:
        matched_id = ids[indices[0][0]]
        server_id = server_ids[indices[0][0]]
        return matched_id, server_id  # Return the ID of the closest match
    else:
        return None,None

def analyze_distances(image_path, index):
    descriptors = extract_features(image_path)
    distances, _ = index.search(descriptors.astype('float32'), 5)  # Search for top 5 matches
    return distances

# Example usage
if __name__ == "__main__":
    # Example image paths
    image_path_1 = 'sample_images/1.jpg'
    image_path_2 = 'sample_images/2.jpg'
    image_path_3 = 'sample_images/3.jpg'
    image_path_4 = 'sample_images/4.jpg'
    image_path_5 = 'sample_images/5.jpg'

#     # Extract and store features for the first image
#     descriptors = extract_features(image_path_1)
#     store_features(descriptors,server_id=random.randint(0,10))

    # Load all descriptors from the database and build the FAISS index
    server_ids, ids, all_descriptors = load_descriptors()
    index = build_faiss_index(all_descriptors)

    distances = analyze_distances(image_path_4, index)
    threshold = np.mean(distances) + np.std(distances)
    print("Threshold set to:", threshold)

    # Search for similar images using the second image
    result,server_id = search_similar(image_path_4, index, ids,server_ids,threshold=threshold)
    print("Matching record ID:", result,server_id)