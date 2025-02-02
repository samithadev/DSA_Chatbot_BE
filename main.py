from fastapi import FastAPI, APIRouter, HTTPException, Depends
from pydantic import BaseModel #data validation
from typing import Optional
from config import pref_collection, user_collection
from database.schemas import individual_data, all_data
from database.models import LearningPreference, User
from bson.objectid import ObjectId
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt


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
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
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


@router.get("/")
async def get_all_prefs():
    data = pref_collection.find() 
    return all_data(data)

@router.get("/pref/{id}")
async def get_pref( id: str):
    data = pref_collection.find_one( {"_id": ObjectId(id)} )
    return individual_data(data)

@router.post("/")
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
    



app.include_router(router)