"""
Excel Answer Key Handler for OMR Processing
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AnswerKeyManager:
    """Manages Excel-based answer keys for OMR evaluation"""

    def __init__(self, answer_keys_dir: str = "data/answer_keys"):
        self.answer_keys_dir = Path(answer_keys_dir)
        self.answer_keys_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Answer key directory: {self.answer_keys_dir}")

    def get_available_versions(self) -> List[str]:
        """Get list of available answer key versions"""
        versions = []
        for file in self.answer_keys_dir.glob("*.xlsx"):
            versions.append(file.stem)
        return sorted(versions)

    def load_answer_key(self, version: str) -> Optional[Dict[int, str]]:
        """Load answer key from Excel file"""
        file_path = self.answer_keys_dir / f"{version}.xlsx"

        if not file_path.exists():
            logger.warning(f"Answer key file not found: {file_path}")
            return None

        try:
            df = pd.read_excel(file_path)
            answer_key = {}

            for _, row in df.iterrows():
                # Handle different column formats
                if 'Question' in row and 'Answer' in row:
                    q_str = str(row['Question']).replace('Q', '').replace('q', '')
                    q_num = int(q_str)
                    answer_key[q_num] = str(row['Answer']).upper()

            logger.info(f"Loaded {len(answer_key)} answers from {version}")
            return answer_key

        except Exception as e:
            logger.error(f"Error loading answer key {version}: {e}")
            return None

    def validate_answer_key(self, version: str) -> Dict[str, any]:
        """Validate answer key format and completeness"""
        answer_key = self.load_answer_key(version)

        if not answer_key:
            return {"valid": False, "error": "Could not load answer key"}

        # Check for 100 questions
        expected_questions = set(range(1, 101))
        actual_questions = set(answer_key.keys())

        missing = expected_questions - actual_questions
        extra = actual_questions - expected_questions

        # Validate answer options
        valid_options = {'A', 'B', 'C', 'D'}
        invalid_answers = {q: ans for q, ans in answer_key.items() 
                          if ans not in valid_options}

        return {
            "valid": len(missing) == 0 and len(invalid_answers) == 0,
            "total_questions": len(answer_key),
            "missing_questions": list(missing),
            "extra_questions": list(extra),
            "invalid_answers": invalid_answers
        }
