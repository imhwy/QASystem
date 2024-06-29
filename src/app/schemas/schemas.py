# schemas.py
from pydantic import BaseModel

class Input(BaseModel):
    context: str
    question: str
    category: str