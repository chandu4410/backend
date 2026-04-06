from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
import psycopg2
import urllib.parse as up
from PIL import Image
import io
import os
from psycopg2 import Binary
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Supabase PostgreSQL connection string from .env
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is not set in environment variables")

result = up.urlparse(DATABASE_URL)

def get_db_connection():
    return psycopg2.connect(
        dbname=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port,
        sslmode='require'
    )

@app.route('/')
def home():
    return "✅ API is running!"

# ---------------------- MATCH ----------------------
@app.route('/api/match', methods=['POST'])
def match_person():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo uploaded'}), 400

    uploaded_file = request.files['photo']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    uploaded_file.stream.seek(0)
    uploaded_image = face_recognition.load_image_file(uploaded_file)
    uploaded_encoding_list = face_recognition.face_encodings(uploaded_image)

    if not uploaded_encoding_list:
        return jsonify({'error': 'No clear face detected'}), 400

    uploaded_encoding = uploaded_encoding_list[0]
    uploaded_encoding_norm = uploaded_encoding / np.linalg.norm(uploaded_encoding)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT person_id, encoding FROM face_encodings")
    rows = cur.fetchall()

    if not rows:
        cur.close()
        conn.close()
        return jsonify({'matchFound': False})

    COSINE_THRESHOLD = 0.85
    DISTANCE_THRESHOLD = 0.6

    best_match = None
    best_cosine = -1

    for person_id, encoding_bytes in rows:
        known_encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
        known_encoding_norm = known_encoding / np.linalg.norm(known_encoding)

        cosine_similarity = np.dot(uploaded_encoding_norm, known_encoding_norm)
        euclidean_distance = np.linalg.norm(uploaded_encoding - known_encoding)

        if cosine_similarity >= COSINE_THRESHOLD and euclidean_distance <= DISTANCE_THRESHOLD:
            if cosine_similarity > best_cosine:
                best_cosine = cosine_similarity
                best_match = (person_id, cosine_similarity, euclidean_distance)

    if best_match:
        match_id, cosine_val, dist_val = best_match
        cur.execute("""
            SELECT name, father_name, phone_number, birth_marks, description 
            FROM person_images WHERE id = %s
        """, (match_id,))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            return jsonify({
                'matchFound': True,
                'name': result[0],
                'fatherName': result[1],
                'phone': result[2],
                'birthMarks': result[3],
                'personalInfo': result[4],
                'cosineSimilarity': round(float(cosine_val * 100), 2),
                'euclideanDistance': round(float(dist_val), 4)
            })

    cur.close()
    conn.close()
    return jsonify({'matchFound': False})

# ---------------------- REGISTER ----------------------
@app.route('/api/register', methods=['POST'])
def register_person():
    data = request.form
    files = request.files.getlist('photos')

    name = data.get('name')
    father_name = data.get('fatherName')
    phone_number = data.get('phone')
    birth_marks = data.get('birthMarks')
    description = data.get('description')

    if not all([name, father_name, phone_number, description]) or not files:
        return jsonify({'error': 'Missing required fields or photos'}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    first_photo = files[0]
    first_photo.stream.seek(0)
    first_image_data = first_photo.read()

    first_photo.stream.seek(0)
    first_image = face_recognition.load_image_file(first_photo)
    first_encodings = face_recognition.face_encodings(first_image)

    if not first_encodings:
        return jsonify({'error': 'No face found in the first image'}), 400

    cur.execute("""
        INSERT INTO person_images (name, father_name, phone_number, birth_marks, description, image)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (name, father_name, phone_number, birth_marks, description, Binary(first_image_data)))
    person_id = cur.fetchone()[0]

    all_encodings = []
    for file in files:
        try:
            file.stream.seek(0)
            image = face_recognition.load_image_file(file)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                enc_norm = encodings[0] / np.linalg.norm(encodings[0])
                all_encodings.append(enc_norm)
        except Exception as e:
            print(f"❌ Skipping file {file.filename}: {e}")

    if not all_encodings:
        return jsonify({'error': 'No faces found in any of the uploaded images'}), 400

    mean_encoding = np.mean(all_encodings, axis=0)
    mean_encoding_norm = mean_encoding / np.linalg.norm(mean_encoding)

    cur.execute(
        "INSERT INTO face_encodings (person_id, encoding) VALUES (%s, %s)",
        (person_id, Binary(mean_encoding_norm.astype(np.float64).tobytes()))
    )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({'success': True, 'person_id': person_id})

# ---------------------- CASES ----------------------
@app.route('/api/cases', methods=['GET'])
def get_all_cases():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, father_name, phone_number, birth_marks, description, image
        FROM person_images
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        image_base64 = base64.b64encode(row[6]).decode('utf-8') if row[6] else None
        results.append({
            'id': row[0],
            'name': row[1],
            'fatherName': row[2],
            'phone': row[3],
            'birthMarks': row[4],
            'personalInfo': row[5],
            'image': image_base64
        })

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
