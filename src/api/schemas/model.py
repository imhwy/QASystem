"""
"""
from pydantic import BaseModel
from typing import Optional


class Models(BaseModel):
    model_id: str
    model_name: str
