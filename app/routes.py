import os
import uuid
import torch
import logging
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from flask import request, current_app
from flask_restx import Namespace, Resource, fields
from werkzeug.utils import secure_filename
from PIL import Image
import io
import google.cloud.storage

# Import from models and configuration
from .models import (
    db, 
    User, 
    SkinAnalysis, 
    ProductRecommendation, 
    SkinAnalysisSchema, 
    ProductRecommendationSchema
)

# Create API Namespace
api = Namespace('SkinAnalysis', description='Endpoints for skin analysis and recommendations')

# Request and response models for Swagger
upload_model = api.model('UploadModel', {
    'user_id': fields.String(description='User ID for the analysis', required=False),
    'image': fields.Raw(description='Image file to analyze (multipart/form-data)', required=True),
})

response_model = api.model('ResponseModel', {
    'skin_type': fields.String(description='Predicted skin type'),
    'confidence': fields.Float(description='Confidence score of the prediction'),
    'image_url': fields.String(description='URL of the uploaded image'),
    'recommendations': fields.List(fields.Raw, description='Product recommendations for the skin type'),
})

recommendations_model = api.model('RecommendationsModel', {
    'id': fields.Integer(description='Product recommendation ID'),
    'skin_type': fields.String(description='Skin type for the recommendation'),
    'product_name': fields.String(description='Name of the recommended product'),
    'description': fields.String(description='Product description'),
    # Add other relevant fields from ProductRecommendation
})

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_image(image_file):
    """
    Preprocess the uploaded image for skin type classification
    """
    try:
        # Open the image
        img = Image.open(image_file).convert('RGB')
        
        # Define transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transformation
        return transform(img).unsqueeze(0)
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

def load_skin_type_model():
    """
    Load the pre-trained skin type classification model
    """
    try:
        model_path = current_app.config['MODEL_PATH']
        
        # Initialize a base model (e.g., ResNet)
        model = models.resnet50(pretrained=False)
        
        # Modify the final layer for skin type classification
        num_classes = 5  # Adjust based on your skin type categories
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def upload_to_gcs(image_file, user_id=None):
    """
    Upload image to Google Cloud Storage
    """
    try:
        # Initialize Google Cloud Storage client
        storage_client = google.cloud.storage.Client.from_service_account_json(
            current_app.config.get('GOOGLE_APPLICATION_CREDENTIALS')
        )
        
        # Get the bucket
        bucket = storage_client.bucket(current_app.config['GCS_BUCKET_NAME'])
        
        # Generate a unique filename
        unique_filename = f"skin_analysis/{user_id or 'anonymous'}/{uuid.uuid4()}.jpg"
        
        # Create a blob and upload
        blob = bucket.blob(unique_filename)
        blob.upload_from_file(image_file)
        
        # Make the blob publicly accessible
        blob.make_public()
        
        return blob.public_url
    except Exception as e:
        logger.error(f"GCS upload error: {e}")
        raise

@api.route('/skin-analysis')
class SkinAnalysisPOST(Resource):
    """Skin Analysis Endpoint"""
    
    @api.doc(description='Upload an image to analyze skin type and get product recommendations.')
    @api.expect(upload_model)
    @api.response(200, 'Success', response_model)
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def post(self):
        """
        Perform skin analysis and get recommendations
        """
        try:
            # Check if image is present
            if 'image' not in request.files:
                return {'error': 'No image uploaded'}, 400
            
            image_file = request.files['image']
            
            # Optional: Get user_id if authenticated
            user_id = request.form.get('user_id')
            
            # Validate file
            if image_file.filename == '':
                return {'error': 'No selected file'}, 400
            
            # Secure the filename
            filename = secure_filename(image_file.filename)
            
            # Preprocess image
            preprocessed_image = preprocess_image(image_file)
            
            # Load model
            model = load_skin_type_model()
            
            # Perform inference
            with torch.no_grad():
                outputs = model(preprocessed_image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted].item()
            
            # Map predicted class to skin type
            skin_types = ['dry', 'oily', 'combination', 'sensitive', 'normal']
            predicted_skin_type = skin_types[predicted.item()]
            
            # Upload image to GCS
            image_file.seek(0)  # Reset file pointer
            image_url = upload_to_gcs(image_file, user_id)
            
            # Save analysis to database
            skin_analysis = SkinAnalysis(
                user_id=user_id,
                skin_type=predicted_skin_type,
                confidence=confidence,
                image_url=image_url
            )
            db.session.add(skin_analysis)
            db.session.commit()
            
            # Get product recommendations
            recommendations = ProductRecommendation.query.filter_by(skin_type=predicted_skin_type).all()
            rec_schema = ProductRecommendationSchema(many=True)
            
            return {
                'skin_type': predicted_skin_type,
                'confidence': float(confidence),
                'image_url': image_url,
                'recommendations': rec_schema.dump(recommendations)
            }, 200
        
        except Exception as e:
            logger.error(f"Skin analysis error: {e}")
            db.session.rollback()
            return {'error': 'Internal server error'}, 500

@api.route('/recommendations')
class ProductRecommendationResource(Resource):
    """Product Recommendations Endpoint"""
    
    @api.doc(description='Get product recommendations, optionally filtered by skin type.')
    @api.param('skin_type', 'Filter recommendations by skin type (optional)')
    @api.response(200, 'Success', recommendations_model)
    @api.response(500, 'Internal Server Error')
    def get(self):
        """
        Get product recommendations
        """
        try:
            # Optional query parameter for skin type
            skin_type = request.args.get('skin_type')
            
            # Query recommendations
            if skin_type:
                recommendations = ProductRecommendation.query.filter_by(skin_type=skin_type).all()
            else:
                recommendations = ProductRecommendation.query.all()
            
            # Serialize recommendations
            rec_schema = ProductRecommendationSchema(many=True)
            return rec_schema.dump(recommendations), 200
        
        except Exception as e:
            logger.error(f"Recommendations retrieval error: {e}")
            return {'error': 'Internal server error'}, 500

@api.route('/user-analyses')
class UserAnalysesResource(Resource):
    """User Skin Analyses Endpoint"""
    
    @api.doc(description='Retrieve a user\'s skin analyses.')
    @api.param('user_id', 'User ID to retrieve analyses for', required=True)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request')
    @api.response(404, 'User Not Found')
    @api.response(500, 'Internal Server Error')
    def get(self):
        """
        Retrieve skin analyses for a specific user
        """
        try:
            # Require user_id as a query parameter
            user_id = request.args.get('user_id')
            if not user_id:
                return {'error': 'User ID is required'}, 400
            
            # Verify user exists
            user = User.query.get(user_id)
            if not user:
                return {'error': 'User not found'}, 404
            
            # Get user's skin analyses
            analyses = SkinAnalysis.query.filter_by(user_id=user_id).order_by(SkinAnalysis.analyzed_at.desc()).all()
            
            # Serialize analyses
            analysis_schema = SkinAnalysisSchema(many=True)
            return analysis_schema.dump(analyses), 200
        
        except Exception as e:
            logger.error(f"User analyses retrieval error: {e}")
            return {'error': 'Internal server error'}, 500

# Error Handlers
@api.errorhandler(413)
def request_entity_too_large(error):
    """
    Handler for file too large errors
    """
    return {'error': 'File too large'}, 413

@api.errorhandler(404)
def not_found(error):
    """
    404 error handler
    """
    return {'error': 'Not found'}, 404

@api.errorhandler(500)
def internal_server_error(error):
    """
    500 error handler
    """
    return {'error': 'Internal server error'}, 500