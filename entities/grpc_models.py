from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional, List

from pydantic import BaseModel


class SuccessState(Enum):
    RESULT_SUCCESS = 0
    RESULT_FAILURE = 1
    RESULT_DENY = 2
    RESULT_UNDEFINED = 3


class FaceIndexes(BaseModel):
    userId: Optional[str] = None
    confidence: float
    label: str
    encoding: List[float]


class User(BaseModel):
    userId: str


class RecentActivity(BaseModel):
    successState: SuccessState = 3
    title: str
    content: str
    causeId: str
    createdTime: datetime = datetime.now()
    modifiedTime: datetime = datetime.now()
