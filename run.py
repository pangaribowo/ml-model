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
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_PATH = 'skin_type.pth'  # Added this line to define MODEL_PATH

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model variable
model = None

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
        print(f"Error loading model weights: {e}")
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
    try:
        # Validate inputs
        if not blob_name or not file_data:
            raise ValueError("Invalid upload parameters")
        
        # Initialize storage client
        storage_client = storage.Client()
        
        # Validate bucket exists
        bucket = storage_client.bucket(BUCKET_NAME)
        if not bucket.exists():
            raise ValueError(f"Bucket {BUCKET_NAME} does not exist")
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            file_data, 
            content_type='image/jpeg',
            # Optional: Add metadata
            metadata={
                'prediction_id': blob_name.split('.')[0],
                'uploaded_at': datetime.utcnow().isoformat()
            }
        )
        
        # Make the blob publicly accessible
        blob.make_public()
        
        app.logger.info(f"Successfully uploaded {blob_name} to {BUCKET_NAME}")
        return blob.public_url
    
    except Exception as e:
        app.logger.error(f"Upload error: {e}")
        raise

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
        # Generate unique ID
        prediction_id = str(uuid.uuid4()).replace('-', '_')
        
        # Read image bytes
        img_bytes = file.read()
        
        # Create unique filename
        file_extension = file.filename.split('.')[-1].lower()
        unique_filename = f"skin_type_prediction_{prediction_id}.{file_extension}"
        
      # Upload to Google Cloud Storage
        try:
            image_url = upload_to_bucket(unique_filename, img_bytes)
        except Exception as upload_error:
            # Log the upload error but don't stop the prediction
            app.logger.error(f"Image upload failed: {upload_error}")
            image_url = None

        # Preprocess and predict
        input_tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_classes = torch.topk(probabilities, 1)
        
        # Skin type labels
        skin_type_labels = ['kering', 'normal', 'berminyak']
        predicted_label = skin_type_labels[top_classes[0].item()]
        confidence_score = top_prob[0].item()
        
        # Prepare response
        response = {
            "message": "Model is predicted successfully.",
            "data": {
                "id": prediction_id,
                "result": predicted_label,
                "confidenceScore": confidence_score,
                "isAboveThreshold": confidence_score > 0.5,
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "imageUrl": image_url  # Add image URL to response
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
    # Load the model before running the app
    model = load_skin_type_model(MODEL_PATH)
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)