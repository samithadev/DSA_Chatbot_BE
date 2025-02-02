from pydantic import BaseModel
from datetime import datetime

class LearningPreference(BaseModel):
    mode: str
    topic: str
    subTopic: str
    level: str
    userId: str
    updated_at: datetime = datetime.now()
    created_at: datetime = datetime.now()


class User(BaseModel):
    username: str
    password: str
    disabled: bool = False