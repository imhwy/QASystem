"""
"""
from pydantic import BaseModel
from typing import Optional


class QuestionContext(BaseModel):
    question: str
    context: str
