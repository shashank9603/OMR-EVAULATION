"""
Innomatics OMR System - Database Initialization
Setup production database with initial data
"""

from sqlalchemy.orm import sessionmaker
from .models import engine, Base, AnswerKey, User
import hashlib
import json

def hash_password(password: str) -> str:
    """Hash password for storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Initialize database with required data"""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Create default admin user
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@innomatics.com",
                password_hash=hash_password("admin123"),
                role="administrator"
            )
            db.add(admin_user)
        
        # Create sample answer keys
        sample_answers_set_a = {str(i): ["A", "B", "C", "D"][i % 4] for i in range(1, 101)}
        sample_subject_mapping = {}
        subjects = ["Python", "EDA", "SQL", "PowerBI", "Statistics"]
        for i in range(1, 101):
            subject_index = (i - 1) // 20
            sample_subject_mapping[str(i)] = subjects[subject_index]
        
        # SET-A Answer Key
        answer_key_a = db.query(AnswerKey).filter(AnswerKey.version == "SET-A").first()
        if not answer_key_a:
            answer_key_a = AnswerKey(
                version="SET-A",
                exam_name="Innomatics Certification Exam",
                answers=sample_answers_set_a,
                subject_mapping=sample_subject_mapping,
                created_by="system"
            )
            db.add(answer_key_a)
        
        db.commit()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
