from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://samitha:xI5fhp3i15AscdRS@cluster0.xgj59.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.chatbot_db

pref_collection = db["learning_pref"]
user_collection = db["users"]
images_collection = db["topic_images"]
conversation_collection = db['conversations']