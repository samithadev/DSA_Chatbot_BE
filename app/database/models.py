from fastapi import File, UploadFile
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

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

class Image(BaseModel):
    imageNo: int
    imageDes: str
    imageUrl: str

class TopicImages(BaseModel):
    topic: str
    pdf: UploadFile = File(...)
    images: List[Image]

# class ChatRequest(BaseModel):
#     learning_mode: str  
#     topic: str
#     sub_topic: str
#     student_level: str
#     user_input: str
#     userId: str 

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class Conversation(BaseModel):
    user_id: str
    preference_id: str
    messages: List[Message]
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class ImageData(BaseModel):
    imageNumber: int
    imageDescription: str
class ChatRequest(BaseModel):
    conversation_id: Optional[str]
    user_id: str
    preference_id:str
    learning_mode: str
    topic: str
    sub_topic: str
    student_level: str
    user_input: str
    relevant_images: List[ImageData]

class Topic(BaseModel):
    topic: str
    status: str