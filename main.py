import os
from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel #data validation
from typing import Optional
from config import pref_collection, user_collection, images_collection, conversation_collection
from database.schemas import individual_data, all_data, all_user_pref_data, images_data
from database.models import LearningPreference, User, TopicImages, ChatRequest, Conversation, Message
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
import google.generativeai as genai


app = FastAPI()
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

origins = [
    "http://localhost:5173",
    "http://localhost:3000"
    # "https://test.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#JWT secret key
SECRET_KEY = "d4e9d2f3a0c9b5f9b1e3f4c2f3b3d1b7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

# User Authentication
@router.post("/register")
def create_user(user: User):
    db_user = user_collection.find_one({"username": user.username})

    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, password=hashed_password)
    res = user_collection.insert_one(dict(new_user))
    return {"message:": "user created sucessfully"}


def login(username: str, password: str):
    user = user_collection.find_one({"username": username})
    if not user:
        return False
    if not pwd_context.verify(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()

    if "_id" in to_encode and isinstance(to_encode["_id"], ObjectId):
        to_encode["_id"] = str(to_encode["_id"])

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/token")
def login_for_access_token(form_data: User):
    user = login(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = str(user['_id']) if isinstance(user['_id'], ObjectId) else user['_id']
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"], "userId":user_id}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=403, detail="Token is invalid or expired")
        return payload
    except JWTError:
        raise HTTPException(status_code=403, detail="Token is invalid or expired")

@router.get("/verify-token/{token}")
async def verify_user_token(token:str):
    verify_token(token= token)
    return {"message": "Token is valid"}


# User preference
@router.get("/")
async def get_all_prefs():
    data = pref_collection.find() 
    return all_data(data)

@router.get("/pref/{id}")
async def get_pref( id: str):
    data = pref_collection.find_one( {"_id": ObjectId(id)} )
    return individual_data(data)

@router.post("/pref")
async def create_pref(new_pref: LearningPreference):
    try:
        new_pref.created_at = datetime.now()
        res = pref_collection.insert_one(dict(new_pref))
        return {"status":200, "id": str(res.inserted_id)}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
@router.put("/{id}")
async def update_pref(id: str, new_pref: LearningPreference):
    try:
        id = ObjectId(id)
        existing_doc = pref_collection.find_one({"_id": id})
        if not existing_doc:
            return HTTPException(status_code=404, detail="No such record found")
        
        new_pref.updated_at = datetime.now()
        res = pref_collection.update_one({"_id": id}, {"$set": dict(new_pref)})
        return {"status":200, "id": str(id), "message": "Record updated successfully"}
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

@router.get("/verify-chat/{id}")
async def verify_chat(id: str):
    data = pref_collection.find_one({"_id": ObjectId(id)})
    if not data:
        raise HTTPException(status_code=400, detail="No such record found")
    return {"message": "Record found"}  

@router.get("/previous_prefs/{userId}")
async def get_user_prefs( userId: str):
    data = pref_collection.find( {"userId": userId} )
    return all_user_pref_data(data)

#handle images
@router.post("/add-images")
async def add_images(new_images: TopicImages):
    try:
        images_dict = [dict(image) for image in new_images.images]

        data = {
            "topic": new_images.topic,
            "images": images_dict
        }

        res = images_collection.insert_one(data)
        return {"status":200, "id": str(res.inserted_id),"message": "Images added successfully!"}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    
@router.get("/topic-images/{topic}")
async def get_images( topic: str):
    data = images_collection.find_one( {"topic": topic} )
    return images_data(data) 

# Configure Gemini API
genai.configure(api_key='AIzaSyDfvIigq68nMiI_Zk8guGMzfRPC1pomdQw')

async def get_gemini_response(prompt: str) -> str:
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Return the text response
        return response.text
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Gemini response: {str(e)}")


# @router.post("/chat")
# async def handle_chat(chat_request: ChatRequest):
#     try:
#         # Fetch or create conversation
#         if chat_request.conversation_id:
#             conversation = conversation_collection.find_one(
#                 {"_id": ObjectId(chat_request.conversation_id)}
#             )
#             if not conversation:
#                 raise HTTPException(status_code=404, detail="Conversation not found")
#         else:
#             # Create new conversation
#             conversation = Conversation(
#                 preference_id= chat_request.preference_id,
#                 user_id=chat_request.user_id,
#                 messages=[]
#             )
#             conversation_collection.insert_one(dict(conversation))
#             conversation = dict(conversation)

#         # Add user message to history
#         new_message = Message(
#             role="user",
#             content=chat_request.user_input
#         )
        
#         # Construct context from conversation history
#         if chat_request.conversation_id:
#             recent_messages = conversation.get('messages', [])[-5:]
#             conversation_context = "\n".join([
#                 f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
#                 for msg in recent_messages
#             ])
#         else:
#             conversation_context = ""
        

#         # Construct prompt with conversation history
#         prompt = f"""
#         You are an expert computer science tutor. Previous conversation context:

#         {conversation_context}

#         Current parameters:
#         Learning Mode: {chat_request.learning_mode}
#         Topic: {chat_request.topic}
#         Sub-topic: {chat_request.sub_topic}
#         Student Level: {chat_request.student_level}
        
#         Guidelines:
#         1. For theory mode:
#            - Focus on conceptual explanations
#            - Explain theoretical foundations
#            - Use appropriate diagrams or examples when needed
#            - Connect concepts to broader computer science principles
        
#         2. For practical mode:
#            - Provide implementation details
#            - Include code examples
#            - Give step-by-step instructions
#            - Share best practices and common pitfalls
        
#         3. Adapt complexity for student level:
#            - Beginner: Use simple terms, basic examples, and avoid jargon
#            - Intermediate: Include technical details and moderate complexity
#            - Expert: Provide in-depth explanations and advanced concepts
        
#         4. Response Format:
#            - Start with a brief overview
#            - Break down complex concepts
#            - Include relevant examples
#            - End with a practice suggestion or next steps
        
#         Current User Question/Input: {chat_request.user_input}
        
#         Provide a well-structured, clear response that matches the learning mode and student level while maintaining context from previous messages.
#         """

#         # Get response from Gemini
#         response = await get_gemini_response(prompt)

#         # Add assistant response to history
#         assistant_message = Message(
#             role="assistant",
#             content=response
#         )

#         # Update conversation in database
#         conversation_collection.update_one(
#             {"_id": ObjectId(conversation['_id'])},
#             {
#                 "$push": {
#                     "messages": {
#                         "$each": [
#                             dict(new_message),
#                             dict(assistant_message)
#                         ]
#                     }
#                 },
#                 "$set": {"updated_at": datetime.now()}
#             }
#         )

#         return {
#             "status": 200,
#             "response": response,
#             "conversation_id": str(conversation['_id']),
#             "message": "Chat response generated successfully"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def handle_chat(chat_request: ChatRequest):
    try:
        conversation = None
        # Handle existing conversation
        if chat_request.conversation_id:
            conversation = conversation_collection.find_one(
                {"_id": ObjectId(chat_request.conversation_id)}
            )
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
        else:
            # Create new conversation
            new_conversation = {
                "preference_id": chat_request.preference_id,
                "user_id": chat_request.user_id,
                "messages": [],
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            # Insert and get the new conversation
            result = conversation_collection.insert_one(new_conversation)
            conversation = {
                "_id": result.inserted_id,
                **new_conversation
            }

        # Add user message to history
        new_message = Message(
            role="user",
            content=chat_request.user_input
        )
        
        # Construct context from conversation history
        recent_messages = conversation.get('messages', [])[-5:]
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in recent_messages
        ])

        # Rest of your prompt construction code remains the same
        # prompt = f"""
        # You are an expert computer science tutor. Previous conversation context:

        # {conversation_context}

        # Current parameters:
        # Learning Mode: {chat_request.learning_mode}
        # Topic: {chat_request.topic}
        # Sub-topic: {chat_request.sub_topic}
        # Student Level: {chat_request.student_level}
        
        # Guidelines:
        # 1. For theory mode:
        #    - Focus on conceptual explanations
        #    - Explain theoretical foundations
        #    - Use appropriate diagrams or examples when needed
        #    - Connect concepts to broader computer science principles
        
        # 2. For practical mode:
        #    - Provide implementation details
        #    - Include code examples
        #    - Give step-by-step instructions
        #    - Share best practices and common pitfalls
        
        # 3. Adapt complexity for student level:
        #    - Beginner: Use simple terms, basic examples, and avoid jargon
        #    - Intermediate: Include technical details and moderate complexity
        #    - Expert: Provide in-depth explanations and advanced concepts
        
        # 4. Response Format:
        #    - Start with a brief overview
        #    - Break down complex concepts
        #    - Include relevant examples
        #    - End with a practice suggestion or next steps
        
        # Current User Question/Input: {chat_request.user_input}
        
        # Provide a well-structured, clear response that matches the learning mode and student level while maintaining context from previous messages.
        # """

        prompt = f"""
            You are an expert computer science tutor focused specifically on {chat_request.topic} and {chat_request.sub_topic}. Adapt your teaching style based on student responses.

            Previous conversation context:
            {conversation_context}

            these are some relevent images for this topic with image number and description.when you give answers if these image descriptions relevent to answers give reference like this. example: (reference: image 1)
            {chat_request.relevant_images}

            Current parameters:
            Learning Mode: {chat_request.learning_mode}
            Topic: {chat_request.topic}
            Sub-topic: {chat_request.sub_topic}
            Student Level: {chat_request.student_level}

            Topic Boundary Guidelines:
            1. Stay strictly within {chat_request.topic} and {chat_request.sub_topic}
            2. If user asks about unrelated topics, respond: "That's outside our current focus on [topic]. Would you like to know more about [specific aspect of current topic]?"
            3. Redirect off-topic questions back to the current learning objectives
            4. Only answer questions relevant to the specified topic and sub-topic
            
            Teaching Guidelines:
            1. Adaptive Response Strategy:
            - Start with a brief conceptual question
            - If student shows knowledge, explore deeper with follow-up questions
            - If student indicates lack of knowledge (e.g., "don't know", "no", "ok"), provide:
                * A clear, concise explanation
                * A simple example
                * A follow-up question to check understanding
            - If student continues to show uncertainty, provide more detailed explanations

            2. Knowledge Assessment:
            - If student shows understanding -> progress to next concept
            - If student shows lack of knowledge -> provide clear explanation with example
            - If student responds with "ok" -> verify understanding with simple question
            
            3. Progressive Teaching:
            - Track concepts student has understood
            - Don't repeat confirmed concepts
            - Move forward once current concept is grasped
            - Build on previously understood concepts
            
            4. Response Structure:
            - Keep initial questions simple and direct
            - When explaining concepts:
                * Start with basic definition
                * Follow with practical example
                * End with simple verification question
            - Provide complete answers when student shows lack of understanding
            
            5. Mode-Specific Guidance:
            Theory Mode:
            - Focus on clear concept explanations
            - Use real-world analogies
            - Provide visual descriptions when helpful
            
            Practical Mode:
            - Show code examples
            - Explain implementation steps
            - Demonstrate practical usage
            
            6. Level-Based Adaptation:
            Beginner: 
            - Provide complete, simple explanations
            - Use basic examples
            - Break down complex concepts
            
            Intermediate:
            - Balance explanation with exploration
            - Use more technical examples
            - Connect to related concepts
            
            Expert:
            - Focus on advanced concepts
            - Discuss optimizations
            - Cover edge cases
            
            7. Conversation Flow:
            - Don't ask multiple questions in succession
            - If student shows lack of knowledge twice, provide complete explanation
            - Keep responses focused and concise
            - Include practical examples in explanations
            
            Current User Input: {chat_request.user_input}
            
            Remember:
            1. Provide direct answers when student shows lack of understanding
            2. Include examples in explanations
            3. Keep responses clear and concise
            4. Stay within topic boundaries
            5. Build on student's current knowledge level
            """

        # Get response from Gemini
        response = await get_gemini_response(prompt)

        # Add assistant response to history
        assistant_message = Message(
            role="assistant",
            content=response
        )

        # Update conversation in database using the conversation ID
        conversation_collection.update_one(
            {"_id": conversation["_id"]},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            dict(new_message),
                            dict(assistant_message)
                        ]
                    }
                },
                "$set": {"updated_at": datetime.now()}
            }
        )

        return {
            "status": 200,
            "response": response,
            "conversation_id": str(conversation["_id"]),
            "message": "Chat response generated successfully"
        }

    except Exception as e:
        print(f"Error in handle_chat: {str(e)}")  # Add debugging
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/conversation/{preference_id}/{user_id}")
async def get_conversation_by_preference(preference_id: str, user_id: str):
    try:
        # Find conversation by preference_id and user_id
        conversation = conversation_collection.find_one({
            "preference_id": preference_id,
            "user_id": user_id
        })
        
        if conversation:
            # Convert ObjectId to string for JSON serialization
            conversation['_id'] = str(conversation['_id'])
            return {"conversation": conversation}
        
        return {"conversation": None}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/deleteconv/{conversationId}")
async def delete_conversation(conversationId: str):
    try:
        # Convert string ID to ObjectId
        result = conversation_collection.find_one_and_delete({"_id": ObjectId(conversationId)})
        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.delete("/deletepref/{preferenceId}")
async def delete_preference(preferenceId: str):  # Fixed function name
    try:
        # Convert string ID to ObjectId
        result = pref_collection.find_one_and_delete({"_id": ObjectId(preferenceId)})
        if not result:
            raise HTTPException(status_code=404, detail="Preference not found")
        return {"message": "Preference deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)