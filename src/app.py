import os
import random
from datetime import datetime
from PIL import Image
import pickle
import torch
import pandas as pd
import requests
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mtcnn = MTCNN(keep_all=True, thresholds=[0.5, 0.6, 0.6], min_face_size=15)
model = InceptionResnetV1(pretrained='vggface2').eval()

def convert_image_format(input_path, output_format='PNG'):
    try:
        image = Image.open(input_path)
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            orientation = exif.get(0x112) if exif else None
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                image = image.rotate(rotations[orientation], expand=True)
        output_path = os.path.splitext(input_path)[0] + '.' + output_format.lower()
        image.convert('RGB').save(output_path, format=output_format)
        return output_path
    except Exception as e:
        logger.error("Error converting image format: %s", e, exc_info=True)
        return None

def load_embeddings(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)

def compare_faces(embedding, dataset, threshold=0.5):
    embedding /= np.linalg.norm(embedding)
    best, name = float('inf'), None
    for person, embeds in dataset.items():
        for known in embeds:
            known = known / np.linalg.norm(known)
            dist = cosine(embedding, known.flatten())
            if dist < best:
                best, name = dist, person
    return name if best < threshold else None

def load_id_mapping(excel_file):
    df = pd.read_excel(excel_file)
    return {row['Name']: row['ID'] for _, row in df.iterrows()}

def process_image(path):
    dataset = load_embeddings('data/face_model.pkl')
    mapping = load_id_mapping('data/test.xlsx')
    path = convert_image_format(path)
    if not path:
        return {'error': 'Conversion failed'}
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        return {'error': str(e)}
    faces, probs = mtcnn(img, return_prob=True)
    results, unknown = [], 0
    if faces is not None:
        for face, p in zip(faces, probs):
            if p < 0.9:
                continue
            face = (face - 0.5) / 0.5
            emb = model(face.unsqueeze(0)).detach().cpu().numpy().flatten()
            person = compare_faces(emb, dataset)
            if person:
                sid = mapping.get(person, 'ID not found')
                logger.info("Matched student - ID: %s, Name: %s", sid, person)
                results.append({'Student Name': person, 'ID': sid})
                status, resp = send_attendance_to_external_api(sid)
                logger.info("Attendance: %s - %s", status, resp)
            else:
                unknown += 1
        if unknown:
            results.append({'unknown_faces': unknown})
    else:
        results.append({'message': 'No faces detected'})
    return results

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces_endpoint():
    logger.info("Received request on /recognize_faces endpoint")
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    filename = f"face{random.randint(1, 1000)}.jpg"
    temp_dir = tempfile.gettempdir()
    full = os.path.join(temp_dir, filename)
    f.save(full)
    logger.info("Saved uploaded file to %s", full)
    return jsonify(process_image(full))

def send_attendance_to_external_api(student_id):
    url = 'http://54.242.19.19:3000/api/attendance'
    rid = random.randint(1, 1000)
    payload = {
        'id': str(rid),
        'studentId': str(student_id),
        'status': 'present',
        'date': datetime.now().strftime('%m/%d/%Y %I:%M:%S %p +00:00')
    }
    logger.info("Sending payload: %s", payload)
    try:
        r = requests.post(url, json=payload)
        try:
            return r.status_code, r.json()
        except Exception as json_err:
            return r.status_code, {'error': 'Invalid JSON', 'raw': r.text}
    except Exception as e:
        logger.error("Attendance error: %s", e, exc_info=True)
        return 500, {'error': str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
