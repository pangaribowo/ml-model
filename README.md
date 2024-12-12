# Skin Type Prediction API

## Project Structure
```
skin-type-prediction/
│
├── main.py                 # Main Flask application
├── requirements.txt        # Python dependencies
├── skin_type.pth           # Pre-trained model weights (not included)
│
├── templates/
│   └── index.html          # Frontend HTML template
│
├── static/
│   └── swagger.json        # Swagger API documentation
│
└── uploads/                # Temporary upload directory
```

## Setup and Installation

1. Clone the repository
2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Set up Google Cloud Storage
   - Create a service account and download the JSON key
   - Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable
   - Update `BUCKET_NAME` in `main.py`

5. Prepare the skin type classification model
   - Place your pre-trained model weights at `skin_type.pth`
   - Ensure the model is compatible with the code structure

## Running the Application

```
python main.py
```

## API Endpoints

- `/`: Main page with prediction interface
- `/predict` (POST): Predict skin type from an uploaded image
- `/docs`: Swagger UI documentation

## Requirements

- Python 3.8+
- PyTorch
- Flask
- Google Cloud Storage client

## Notes

- Maximum file upload size: 1MB
- Supported file types: PNG, JPG, JPEG