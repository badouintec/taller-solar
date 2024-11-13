from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from skimage.io import ImageCollection
import os
import json
import random

app = Flask(__name__)

# Ruta de las imágenes térmicas
IMAGES_PATH = 'images'
# Directorio para guardar los archivos JSON
LABELS_PATH = 'labels'
# Directorio para las imágenes estáticas
STATIC_IMAGES_PATH = 'static/images'

# Asegurarse de que el directorio de etiquetas y de imágenes estáticas existan
os.makedirs(LABELS_PATH, exist_ok=True)
os.makedirs(STATIC_IMAGES_PATH, exist_ok=True)

# Cargar las imágenes
images = ImageCollection(os.path.join(IMAGES_PATH, '*.tif'))

# Mantener un registro de las imágenes ya etiquetadas
labeled_images = set()

# Cargar las imágenes ya etiquetadas desde los archivos JSON
for filename in os.listdir(LABELS_PATH):
    if filename.endswith('.json'):
        with open(os.path.join(LABELS_PATH, filename), 'r') as f:
            data = json.load(f)
            labeled_images.add(data['original_image_path'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image')
def get_image():
    # Filtrar las imágenes ya etiquetadas
    available_images = [img for img in images.files if img not in labeled_images]
    
    if not available_images:
        return jsonify({'image_path': None, 'image_idx': None})

    image_idx = random.randint(0, len(available_images) - 1)
    image_path = available_images[image_idx]
    labeled_images.add(image_path)
    thermal_image = cv2.normalize(images[image_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    static_image_path = os.path.join(STATIC_IMAGES_PATH, os.path.basename(image_path))
    cv2.imwrite(static_image_path, thermal_image)

    return jsonify({'image_path': '/' + static_image_path, 'image_idx': image_idx, 'original_image_path': image_path})

@app.route('/save_hotspot', methods=['POST'])
def save_hotspot():
    data = request.get_json()
    image_idx = data['image_idx']
    original_image_path = data['original_image_path']

    if not os.path.exists(LABELS_PATH):
        os.makedirs(LABELS_PATH)

    label_data = {
        'image_idx': image_idx,
        'original_image_path': original_image_path,
        'labels': data['labels']
    }

    filename = os.path.join(LABELS_PATH, f"{os.path.basename(original_image_path).split('.')[0]}.json")
    with open(filename, 'w') as f:
        json.dump(label_data, f)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)