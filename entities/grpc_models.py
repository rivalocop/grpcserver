from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class FaceIndexes(BaseModel):
    userId: Optional[str] = None
    confidence: float
    label: str
    encoding: List[float]


class User(BaseModel):
    userId: str


class RecentActivity(BaseModel):
    isSuccess: bool = False
    title: str
    content: str
    causeId: str
    createdTime: datetime = datetime.now()
    modifiedTime: datetime = datetime.now()
