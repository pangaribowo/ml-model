import os
import logging
from flask import Flask
from flask_migrate import Migrate
from flask_restx import Api

def create_app(config_object=None):
    """
    Application factory function to create and configure Flask app
    """
    # Create Flask app instance
    app = Flask(__name__)

    # Load configuration
    from .config import get_config
    config_class = get_config() if config_object is None else config_object
    app.config.from_object(config_class)

    # Configure logging
    configure_logging(app)

    # Initialize extensions
    from .models import db, ma
    db.init_app(app)
    ma.init_app(app)

    # Initialize Flask-Migrate
    migrate = Migrate(app, db)

    # Register API documentation using Flask-RESTX
    from .routes import api as skin_analysis_ns
    rest_api = Api(app, version='1.0', title='Cekulit API',
                   description='API Documentation for Cekulit Skin Analysis',
                   doc='/docs')
    rest_api.add_namespace(skin_analysis_ns, path='/api/v1')

    return app

def configure_logging(app):
    """
    Configure application logging
    """
    # Remove default Flask logger
    app.logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(app.config['LOG_LEVEL'])

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to app logger
    app.logger.addHandler(console_handler)
    app.logger.setLevel(app.config['LOG_LEVEL'])