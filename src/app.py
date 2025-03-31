import os
from PIL import Image
import pickle
import torch
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True, thresholds=[0.5, 0.6, 0.6], min_face_size=15)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Function to convert image format
def convert_image_format(input_path, output_format='PNG'):
    try:
        image = Image.open(input_path)
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif and 0x112 in exif:
                orientation = exif[0x112]
                rotations = {3: 180, 6: 270, 8: 90}
                if orientation in rotations:
                    image = image.rotate(rotations[orientation], expand=True)
        output_path = os.path.splitext(input_path)[0] + '.' + output_format.lower()
        image.convert("RGB").save(output_path, format=output_format)
        return output_path
    except Exception as e:
        return None

# Load embeddings dataset
def load_embeddings(model_file):
    with open(model_file, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

# Compare embeddings with the dataset
def compare_faces(embedding, dataset, threshold=0.5):
    embedding = embedding / np.linalg.norm(embedding)
    min_distance = float('inf')
    identified_name = None
    for name, embeddings in dataset.items():
        for known_embedding in embeddings:
            known_embedding = known_embedding / np.linalg.norm(known_embedding)
            distance = cosine(embedding, known_embedding.flatten())
            if distance < min_distance:
                min_distance = distance
                identified_name = name
    return identified_name if min_distance < threshold else None

# Load ID mapping
def load_id_mapping(excel_file):
    df = pd.read_excel(excel_file)
    return {row['Name']: row['ID'] for _, row in df.iterrows()}

# Process the image
def process_image(file_path):
    dataset = load_embeddings('data/face_model.pkl')
    name_to_id = load_id_mapping('data/test.xlsx')

    file_path = convert_image_format(file_path, 'PNG')
    if not file_path:
        return {"error": "Failed to convert image to PNG format."}

    try:
        image = Image.open(file_path).convert('RGB')
    except Exception as e:
        return {"error": str(e)}

    faces, probs = mtcnn(image, return_prob=True)
    results = []
    unknown_count = 0

    if faces is not None:
        for face, prob in zip(faces, probs):
            if prob < 0.90:
                continue
            face = (face - 0.5) / 0.5
            embedding = model(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            name = compare_faces(embedding, dataset, threshold=0.5)
            if name:
                person_id = name_to_id.get(name, "ID not found")
                results.append({"Student Name": name, "ID": person_id})
                print(f"Student Name: {name}, ID: {person_id}")  
            else:
                unknown_count += 1
        if unknown_count > 0:
            results.append({"unknown_faces": unknown_count})
            print(f"Unknown faces: {unknown_count}")  
    else:
        results.append({"message": "No faces detected."})
        print("No faces detected.")  

    return results

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    result = process_image(file_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
