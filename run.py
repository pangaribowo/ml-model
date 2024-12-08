import os
import io
import uuid
from datetime import datetime
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from google.cloud import storage
from flask import Flask, request, jsonify, render_template, send_file
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename

# Configuration
BUCKET_NAME = 'your-bucket-name'
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model globally
MODEL_PATH = 'skin_type.pth'
model = None

def load_skin_type_model(model_path, num_classes=3):
    """
    Load the pre-trained skin type classification model
    
    Args:
        model_path (str): Path to the model weights
        num_classes (int): Number of skin type classes
    
    Returns:
        torch.nn.Module: Loaded and configured model
    """
    global model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load the model
model = load_skin_type_model(MODEL_PATH)

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

def preprocess_image(image_bytes):
    """
    Preprocess the input image for model prediction
    
    Args:
        image_bytes (bytes): Image data in bytes
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def upload_to_bucket(blob_name, file_data):
    """
    Upload image to Google Cloud Storage
    
    Args:
        blob_name (str): Name of the blob in the bucket
        file_data (bytes): Image data to upload
    
    Returns:
        str: Public URL of the uploaded image
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(file_data, content_type='image/jpeg')
    return blob.public_url

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
            "message": "File size exceeds 1MB limit.",
            "data": None
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            "message": "Invalid file type. Only JPG, JPEG, and PNG are allowed.",
            "data": None
        }), 400
    
    try:
        # Read image bytes
        img_bytes = file.read()
        
        # Preprocess and predict
        input_tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_classes = torch.topk(probabilities, 1)
        
        # Skin type labels
        skin_type_labels = ['dry', 'normal', 'oily']
        predicted_label = skin_type_labels[top_classes[0].item()]
        confidence_score = top_prob[0].item()
        
        # Generate unique ID
        prediction_id = str(uuid.uuid4()).replace('-', '_')
        
        # Prepare response
        response = {
            "message": "Model is predicted successfully.",
            "data": {
                "id": prediction_id,
                "result": predicted_label,
                "confidenceScore": confidence_score,
                "isAboveThreshold": confidence_score > 0.5,
                "createdAt": datetime.utcnow().isoformat() + "Z"
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
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)