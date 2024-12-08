import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app import create_app
from app.models import db

app = create_app()

if __name__ == '__main__':
    with app.app_context():
        # Ensure database is created
        db.create_all()
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)