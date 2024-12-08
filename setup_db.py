import os
import sys
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models import db, User, SkinAnalysis, ProductRecommendation
from app.config import Config

def setup_database():
    """
    Create database and tables if they don't exist
    """
    # Load environment variables
    load_dotenv()

    # Get database URI
    db_uri = Config.SQLALCHEMY_DATABASE_URI

    # Create engine
    engine = create_engine(db_uri)

    # Create database if not exists
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Database created: {db_uri}")

    # Create tables
    from app import create_app
    with create_app().app_context():
        db.create_all()
        print("Tables created successfully!")

def seed_initial_data():
    """
    Seed initial product recommendations
    """
    from app import create_app
    app = create_app()

    with app.app_context():
        # Check if recommendations already exist
        existing_recommendations = ProductRecommendation.query.count()
        if existing_recommendations > 0:
            print("Product recommendations already exist. Skipping seeding.")
            return

        # Initial product recommendations
        initial_recommendations = [
            ProductRecommendation(
                name="Moisturizer for Dry Skin",
                description="Hydrating cream for extremely dry skin",
                skin_type="dry",
                brand="HydraSkin",
                category="Moisturizer",
                price=29.99,
                product_url="https://example.com/dry-skin-moisturizer"
            ),
            ProductRecommendation(
                name="Oil Control Serum",
                description="Lightweight serum for oily skin control",
                skin_type="oily",
                brand="MatteEffect",
                category="Serum",
                price=35.50,
                product_url="https://example.com/oily-skin-serum"
            ),
            # Add more recommendations for different skin types
        ]

        # Add to database
        db.session.bulk_save_objects(initial_recommendations)
        db.session.commit()
        print("Initial product recommendations seeded.")

if __name__ == "__main__":
    setup_database()
    seed_initial_data()