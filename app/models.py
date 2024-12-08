from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow import fields
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

# Initialize extensions
db = SQLAlchemy()
ma = Marshmallow()

class User(db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    skin_analyses = relationship('SkinAnalysis', back_populates='user')

    def __repr__(self):
        return f'<User {self.username}>'

class SkinAnalysis(db.Model):
    """Skin type analysis result model"""
    __tablename__ = 'skin_analyses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    skin_type = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    image_url = Column(String(255), nullable=False)
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship('User', back_populates='skin_analyses')

    def __repr__(self):
        return f'<SkinAnalysis {self.skin_type} - {self.confidence}>'

class ProductRecommendation(db.Model):
    """Product recommendation model"""
    __tablename__ = 'product_recommendations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    skin_type = Column(String(20), nullable=False)
    product_url = Column(String(255), nullable=True)
    brand = Column(String(50), nullable=True)
    category = Column(String(50), nullable=True)
    price = Column(Float, nullable=True)

    def __repr__(self):
        return f'<ProductRecommendation {self.name} for {self.skin_type}>'

# Marshmallow Schemas for Serialization
class UserSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = User
        load_instance = True
        exclude = ('password_hash',)

class SkinAnalysisSchema(ma.SQLAlchemyAutoSchema):
    user = fields.Nested(UserSchema, exclude=('skin_analyses',))

    class Meta:
        model = SkinAnalysis
        load_instance = True

class ProductRecommendationSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = ProductRecommendation
        load_instance = True