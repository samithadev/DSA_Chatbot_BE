import asyncio
import json
import os
import shutil
import uuid
from fastapi import Body, FastAPI, APIRouter, File, Form, HTTPException, Depends, Path, UploadFile
from pydantic import BaseModel #data validation
from typing import Dict, List, Optional
from config import pref_collection, user_collection, images_collection, conversation_collection, topic_status_collection
from database.schemas import individual_data, all_data, all_user_pref_data, images_data, topic_status
from database.models import LearningPreference, User, TopicImages, ChatRequest, Conversation, Message
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
import google.generativeai as genai

# langchain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader


app = FastAPI()
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# origins = [
#     "http://localhost:5173",
#     "http://localhost:3000"
#     # "https://test.com"
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#JWT secret key
SECRET_KEY = "d4e9d2f3a0c9b5f9b1e3f4c2f3b3d1b7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

class AgenticRAG:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Small, fast, and effective model
        )
        self.vector_store = None
        # Persist Chroma database to disk
        self.vector_store = Chroma(
            embedding_function=self.embeddings
        )
    
    # async def clear_vector_db(self):
    #     """
    #     Deletes all data in the Chroma vector database by removing the persistent directory.
    #     """
    #     try:
    #         if os.path.exists("./chroma_db"):
    #             await asyncio.to_thread(shutil.rmtree, "./chroma_db")
    #         self.vector_store = Chroma(
    #             persist_directory="./chroma_db",
    #             embedding_function=self.embeddings
    #         )
    #         return {"message": "Vector database cleared successfully."}
    #     except Exception as e:
    #         return {"error": str(e)}
    
    async def clear_vector_db(self):
        """
        Completely clears the Chroma vector database by removing the persistent directory
        and reinitializing the vector store.
        """
        try:
            # Close the current vector store client to release file handles
            if self.vector_store:
                ids = self.vector_store.get()['ids']
                self.vector_store.delete(ids)
                
            return {"message": "Vector database completely cleared ids: {ids}"}
        except Exception as e:
            return {"error": f"Failed to clear vector database: {str(e)}"}
    
    async def is_initialized(self):
        """Check if the vector store has been initialized and verify topics"""
        try:
            
            # Check if vector store exists and is initialized
            if not self.vector_store or not hasattr(self.vector_store, "_collection"):
                return {
                    "status": "uninitialized",
                    "message": "Vector store is not initialized"
                }
            
            topics_cursor = images_collection.distinct("topic")
            all_topics = list(topics_cursor)  # Convert to list

            if not all_topics:
                return {"status": "error", "message": "No topics found in database"}

            # Get all topics from the vector store
            vector_topics = set()
            for doc in self.vector_store._collection.get()["metadatas"]:
                if "topic" in doc:
                    vector_topics.add(doc["topic"])

            # Find missing topics
            missing_topics = [topic for topic in all_topics if topic not in vector_topics]

            if missing_topics:
                return {"status": "partial", "available_topics": list(vector_topics), "missing_topics": missing_topics}
            else:
                return {"status": "initialized", "available_topics": all_topics}

        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def process_pdf(self, file_path: str, topic: str):
        """Process PDF and store chunks in vector database"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split text into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata['topic'] = topic
            
            # Create or update vector store
            if not self.vector_store:
                self.vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
            else:
                self.vector_store.add_documents(chunks)
            
            return {"status": "success", "message": f"Processed {len(chunks)} chunks"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def retrieve_relevant_content(self, query: str, topic: str, k: int = 3) -> str:
        """Retrieve relevant content based on query and topic"""
        if not self.vector_store:
            return ""
        
        # Create metadata filter for topic
        filter_dict = {"topic": topic}
        
        # Search for relevant documents
        docs = self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )
        
        # Combine relevant content
        relevant_content = "\n\n".join([doc.page_content for doc in docs])
        return relevant_content

rag_system = AgenticRAG()

@router.get("/check-rag-status")
async def check_rag_status():
    is_init = await rag_system.is_initialized()
    return {"initialized": is_init}

# Add these routes to your existing router
@router.post("/initialize-rag")
async def initialize_rag():

    init_status = await rag_system.is_initialized()
    
    if init_status["status"] == "initialized":
        return {"message": "RAG system already fully initialized"}
    
    pdf_dir = "./pdfs"  # Update with your PDF directory
    topics_to_initialize = init_status.get("missing_topics")
    
    
    results = []
    for topic in topics_to_initialize:
        pdf_path = os.path.join(pdf_dir, f"{topic}.pdf")
        if os.path.exists(pdf_path):
            result = await rag_system.process_pdf(pdf_path, topic)
            results.append({topic: result})
        else:
            results.append({topic: "PDF not found"})
    
    return {"message": "RAG system initialized", "results": results}

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

@router.get("/all-topics")
async def get_all_topics():
    data = images_collection.find()
    topics =  [d["topic"] for d in data]
    return topics

PDF_STORAGE_DIR = "pdfs"
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

@router.post("/add-images")
async def add_images(
    topic: str = Form(...),  # Required topic
    images: Optional[str] = Form(None),  # Optional images (JSON string)
    pdf: Optional[UploadFile] = File(default=None)  # Optional PDF upload
):
    try:
        # Parse images JSON if provided, else default to an empty list
        images_list = json.loads(images) if images else []

        # Check if topic exists in DB
        existing_record = images_collection.find_one({"topic": topic})

        if existing_record and pdf:
            pdf_filename = topic.join([uuid.uuid4().hex, ".pdf"])
        else:
            pdf_filename = None

        if pdf:
            # pdf_filename = pdf.filenamepdf.filename
            if pdf.filename == "":  # Check if empty file is uploaded
                pdf_filename = None 
            else:
                file_extension = os.path.splitext(pdf.filename)[1]  # Gets extension with dot (.pdf)
            
                # Use topic name as the filename
                pdf_filename = f"{topic}{file_extension}"

                pdf_path = os.path.join(PDF_STORAGE_DIR, pdf_filename)

            if os.path.exists(pdf_path):
                return {"status": False, "message": f"PDF file '{pdf_filename}' already exists"}

            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf.file, buffer)

            pdf.file.seek(0)

        if existing_record:
            if images_list:
                existing_images = existing_record.get("images", [])
                max_image_no = max((img["imageNo"] for img in existing_images), default=0)

                new_images = [
                    {"imageNo": max_image_no + i + 1, "imageDes": img["imageDes"], "imageUrl": img["imageUrl"]}
                    for i, img in enumerate(images_list)
                ]

                images_collection.update_one(
                    {"topic": topic},
                    {"$push": {"images": {"$each": new_images}}}
                )

            return {"status": 200, "message": "PDF and images (if any) added successfully to existing topic!"}

        else:
            new_data = {
                "topic": topic,
                "images": [
                    {"imageNo": i + 1, "imageDes": img["imageDes"], "imageUrl": img["imageUrl"]}
                    for i, img in enumerate(images_list)
                ] if images_list else [],
                "pdf": pdf_filename if pdf else None
            }
            res = images_collection.insert_one(new_data)

            return {
                "status": 200,
                "id": str(res.inserted_id),
                "message": "PDF and images (if any) added successfully!"
            }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for images")

    except asyncio.CancelledError:
        print("Request was cancelled by the client or server!")
        raise HTTPException(status_code=499, detail="Request cancelled")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
     
@router.get("/topic-images/{topic}")
async def get_images( topic: str):
    data = images_collection.find_one( {"topic": topic} )
    return images_data(data) 

# Configure Gemini API
genai.configure(api_key='AIzaSyDfvIigq68nMiI_Zk8guGMzfRPC1pomdQw')

async def get_gemini_response(prompt: str) -> str:
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Return the text response
        return response.text
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Gemini response: {str(e)}")


async def optimized_relevant_content(user_input, relevant_content):

    prompt = f""" 
        User Query: {user_input}

        Given Reference Material:
        {relevant_content}

        Task:
        - Extract only the most relevant details from the reference material that directly answer the user query.
        - Do not add any external information; only use the given reference material.
        - Summarize the extracted content concisely while keeping key details intact.
        - Ensure clarity and coherence in the final output.

        Output:
        Provide a structured and concise summary of the most relevant information.
    """
    return await get_gemini_response(prompt)

@router.delete("/vector-db/clear")
async def clear_db():
    """
    Deletes all collections from the vector database.
    """
    try:
        # Assuming `vector_store` has a method to list and delete collections
        await rag_system.clear_vector_db()
        return {"message": "All collections deleted successfully."}
    except Exception as e:
        return {"error": str(e)}

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
        recent_messages = conversation.get('messages', [])[-3:]
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in recent_messages
        ])

        relevant_content = await rag_system.retrieve_relevant_content(
            query=chat_request.user_input,
            topic=chat_request.topic.lower()
        )

        reference_content = await optimized_relevant_content(chat_request.user_input, relevant_content)

        prompt = f"""
            You are an expert computer science tutor focused specifically on {chat_request.topic} and {chat_request.sub_topic}. Adapt your teaching style based on student responses.

            Previous conversation context:
            {conversation_context}

            Relevant reference material:
            {reference_content}

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
            - if no any programming language specified use only java programming language
            
            Practical Mode:
            - Show code examples
            - Explain implementation steps
            - Demonstrate practical usage
            - if no any programming language specified use only java programming language
            
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
            "promt": prompt,
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