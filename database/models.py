"""
Innomatics OMR System - Production Database Models
SQLAlchemy models for production data storage
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///innomatics_omr.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class OMRSheet(Base):
    """OMR Sheet processing records"""
    __tablename__ = "omr_sheets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sheet_id = Column(String, unique=True, nullable=False)
    student_id = Column(String, nullable=True)
    exam_id = Column(String, nullable=False)
    sheet_version = Column(String, nullable=False)  # SET-A, SET-B, etc.
    
    # Processing metadata
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Results
    detected_answers = Column(JSON, nullable=True)  # {question_num: option}
    answer_confidences = Column(JSON, nullable=True)  # {question_num: confidence}
    subject_scores = Column(JSON, nullable=True)  # {subject_num: {correct: int, total: int}}
    total_score = Column(Integer, default=0)
    percentage_score = Column(Float, default=0.0)
    
    # Quality metrics
    overall_confidence = Column(Float, default=0.0)
    image_quality_score = Column(Float, default=0.0)
    requires_review = Column(Boolean, default=False)
    
    # Status
    processing_status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    errors = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)

class AnswerKey(Base):
    """Answer keys for different exam versions"""
    __tablename__ = "answer_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    version = Column(String, unique=True, nullable=False)  # SET-A, SET-B, etc.
    exam_name = Column(String, nullable=False)
    
    # Answer key data
    answers = Column(JSON, nullable=False)  # {question_num: correct_option}
    subject_mapping = Column(JSON, nullable=False)  # {question_num: subject_name}
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

class ProcessingJob(Base):
    """Batch processing jobs"""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_name = Column(String, nullable=False)
    exam_id = Column(String, nullable=False)
    
    # Job details
    total_sheets = Column(Integer, default=0)
    processed_sheets = Column(Integer, default=0)
    failed_sheets = Column(Integer, default=0)
    
    # Status
    status = Column(String, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results summary
    average_score = Column(Float, nullable=True)
    average_confidence = Column(Float, nullable=True)
    sheets_requiring_review = Column(Integer, default=0)

class User(Base):
    """User management"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    
    # Authentication
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)  # administrator, evaluator, coordinator, mentor
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)
