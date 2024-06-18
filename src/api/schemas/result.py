"""
"""
from pydantic import BaseModel
from typing import Optional


class AnswerReponse(BaseModel):
    info_id: str
    context: str
    question: str
    answer: str
