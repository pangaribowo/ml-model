import os
import io
import uuid
import hashlib
import threading
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from flask import Flask, request, jsonify, render_template
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename
from google.cloud import storage
from google.oauth2 import service_account

# Configuration
BUCKET_NAME = 'your-bucket-name'
# UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_PATH = 'skin_type.pth'

# Initialize Flask app
app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model and thread pool variables
model = None
thread_pool = ThreadPoolExecutor(max_workers=2)
prediction_lock = threading.Lock()

def load_skin_type_model(model_path, num_classes=3):
    """
    Load and customize a pre-trained ResNet model for skin type classification
    
    Args:
        model_path (str): Path to the model weights
        num_classes (int): Number of skin type classes
    
    Returns:
        torch.nn.Module: Configured and loaded model
    """
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=False)
    
    # Freeze base model layers (optional, but can help prevent overfitting)
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout for regularization
        nn.Linear(num_features, num_classes)
    )
    
    # Load the state dictionary
    try:
        # Load state dict with CPU mapping and custom logic to handle mismatched keys
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Remove any keys that don't match the current model
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        
        # Partially load the state dict
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        app.logger.error(f"Error loading model weights: {e}")
        # Reinitialize the final layer if weight loading fails
        nn.init.xavier_uniform_(model.fc[1].weight)
        nn.init.zeros_(model.fc[1].bias)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def allowed_file(filename):
    """
    Check if the file extension is allowed
    
    Args:
        filename (str): Name of the file to check
    
    Returns:
        bool: True if file is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes, max_size=1024*1024):
    """
    Enhanced image preprocessing with integrity validation
    
    Args:
        image_bytes (bytes): Image data
        max_size (int): Maximum allowed image size
    
    Returns:
        dict: Processed image tensor and hash
    """
    # Validate image size
    if len(image_bytes) > max_size:
        raise ValueError("Image size exceeds maximum allowed limit")

    # Generate image hash for integrity checking
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Open and convert image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    processed_tensor = transform(image).unsqueeze(0)
    
    return {
        'tensor': processed_tensor,
        'hash': image_hash
    }

def compress_image(image_bytes, quality=85):
    """
    Smart image compression with quality preservation
    
    Args:
        image_bytes (bytes): Original image data
        quality (int): Compression quality
    
    Returns:
        dict: Compressed image bytes and compression ratio
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Convert to RGB to ensure compatibility
        img = img.convert('RGB')
        
        # Compression buffer
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        
        compressed_bytes = output.getvalue()
        compression_ratio = len(compressed_bytes) / len(image_bytes)
        
        return {
            'bytes': compressed_bytes,
            'compression_ratio': compression_ratio
        }

def validate_image_integrity(original_bytes, processed_bytes):
    """
    Validate image integrity after compression
    
    Args:
        original_bytes (bytes): Original image data
        processed_bytes (bytes): Processed image data
    
    Returns:
        dict: Integrity validation results
    """
    original_hash = hashlib.md5(original_bytes).hexdigest()
    processed_hash = hashlib.md5(processed_bytes).hexdigest()
    
    # Convert images to numpy arrays for SSIM calculation
    original_img = np.array(Image.open(io.BytesIO(original_bytes)))
    processed_img = np.array(Image.open(io.BytesIO(processed_bytes)))
    
    ssim_score = ssim(
        original_img,
        processed_img,
        multichannel=True
    )
    
    return {
        'hash_match': original_hash == processed_hash,
        'ssim_score': ssim_score,
        'is_valid': ssim_score > 0.95
    }

def upload_to_bucket(blob_name, file_data):
    """
    Upload image to Google Cloud Storage
    
    Args:
        blob_name (str): Name of the blob in the bucket
        file_data (bytes): Image data to upload
    
    Returns:
        str: Public URL of the uploaded image
    """
    try:
        # Explicitly load credentials from JSON file
        credentials = service_account.Credentials.from_service_account_file(
            'service.json'
        )

        # Initialize storage client with credentials
        storage_client = storage.Client(credentials=credentials)

        # Get bucket
        bucket = storage_client.bucket(BUCKET_NAME)

        # Create blob
        blob = bucket.blob(blob_name)

        # Upload with metadata
        blob.metadata = {
            'prediction_id': blob_name.split('.')[0],
            'uploaded_at': datetime.utcnow().isoformat()
        }

        # Upload file
        blob.upload_from_string(
            file_data,
            content_type='image/jpeg'
        )

        # Make public (optional)
        blob.make_public()

        return blob.public_url

    except Exception as e:
        app.logger.error(f"Detailed upload error: {e}")
        raise

def prediction_with_fallback(image_bytes):
    """
    Fallback strategy for handling processing failures
    
    Args:
        image_bytes (bytes): Image data to process
    
    Returns:
        dict: Prediction results or error information
    """
    try:
        # Primary processing attempt
        return process_prediction(image_bytes)
    
    except Exception as primary_error:
        app.logger.warning(f"Primary prediction failed: {primary_error}")
        
        try:
            # Try with compressed image
            compressed_image = compress_image(image_bytes, quality=75)
            return process_prediction(compressed_image['bytes'])
        
        except Exception as secondary_error:
            app.logger.error(f"Secondary prediction failed: {secondary_error}")
            return {
                'status': 'error',
                'message': 'Prediction failed',
                'details': str(secondary_error)
            }

def process_prediction(image_bytes):
    """
    Process image for skin type prediction
    
    Args:
        image_bytes (bytes): Image data to process
    
    Returns:
        dict: Prediction results
    """
    with prediction_lock:
        # Preprocess image
        processed_img = preprocess_image(image_bytes)
        
        # Run prediction
        with torch.no_grad():
            outputs = model(processed_img['tensor'])
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_classes = torch.topk(probabilities, 1)
        
        return {
            'probabilities': probabilities,
            'top_prob': top_prob,
            'top_classes': top_classes,
            'image_hash': processed_img['hash']
        }

@app.route('/')
def index():
    """
    Render the main landing page
    
    Returns:
        str: Rendered HTML template
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict skin type based on uploaded image
    
    Returns:
        json: Prediction results
    """
    if 'photo' not in request.files:
        return jsonify({
            "message": "No file uploaded.",
            "data": None
        }), 400
    
    file = request.files['photo']
    
    # Validate file
    if file.filename == '':
        return jsonify({
            "message": "No selected file.",
            "data": None
        }), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({
            "message": "File size exceeds 10MB limit.",
            "data": None
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed.",
            "data": None
        }), 400
    
    try:
        # Generate unique ID
        prediction_id = str(uuid.uuid4()).replace('-', '_')
        
        # Read image bytes
        img_bytes = file.read()
        
        # Parallel processing function
        def upload_to_storage():
            try:
                # Create unique filename
                file_extension = file.filename.split('.')[-1].lower()
                unique_filename = f"skin_type_prediction_{prediction_id}.{file_extension}"
                
                # Upload to Google Cloud Storage
                return upload_to_bucket(unique_filename, img_bytes)
            except Exception as upload_error:
                app.logger.error(f"Image upload failed: {upload_error}")
                return None

        # Submit tasks to thread pool
        prediction_result_future = thread_pool.submit(prediction_with_fallback, img_bytes)
        storage_future = thread_pool.submit(upload_to_storage)

        # Wait for both tasks to complete
        prediction_result = prediction_result_future.result()
        image_url = storage_future.result()

        # Check if prediction was successful
        if prediction_result.get('status') == 'error':
            return jsonify({
                "message": prediction_result['message'],
                "details": prediction_result['details'],
                "data": None
            }), 500

        # Compression and validation (optional step)
        try:
            compressed_img = compress_image(img_bytes)
            integrity_check = validate_image_integrity(img_bytes, compressed_img['bytes'])
            app.logger.info(f"Compression integrity: {integrity_check}")
        except Exception as compression_error:
            app.logger.warning(f"Compression validation failed: {compression_error}")

        # Skin type descriptions
        descriptions = {
            'kering': 'Kulit Anda tergolong kering. Kulit kering cenderung kurang kelembapan sehingga dapat terlihat kasar atau bersisik, serta sering terasa kencang, terutama setelah mencuci wajah. Penting untuk menjaga hidrasi kulit dengan pelembap yang sesuai.',
            'normal': 'Kulit Anda tergolong normal. Kulit normal memiliki keseimbangan kadar minyak dan kelembapan yang ideal, dengan tekstur yang lembut serta jarang mengalami masalah kulit seperti jerawat atau kemerahan.',
            'berminyak': 'Kulit Anda tergolong berminyak. Kulit berminyak ditandai dengan produksi sebum (minyak alami kulit) yang berlebihan, yang dapat menyebabkan tampilan mengkilap atau berminyak, terutama di daerah T (dahi, hidung, dan dagu).'
        }

        # Skin type labels
        skin_type_labels = ['kering', 'normal', 'berminyak']
        predicted_label = skin_type_labels[prediction_result['top_classes'][0].item()]
        confidence_score = prediction_result['top_prob'][0].item()
        
        # Prepare response
        response = {
            "message": "Model predicted successfully.",
            "data": {
                "id": prediction_id,
                "result": predicted_label,
                "confidenceScore": confidence_score,
                "isAboveThreshold": confidence_score > 0.5,
                "description": descriptions[predicted_label],
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "imageUrl": image_url,
                "imageHash": prediction_result['image_hash']
            }
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "message": f"Prediction error: {str(e)}",
            "data": None
        }), 500

# Swagger UI Configuration
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Skin Type Prediction API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == '__main__':
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load the model before running the app
    model = load_skin_type_model(MODEL_PATH)
    
    # Ensure upload folder exists
    # os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)