import os
import sys
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# --- 1. SETUP & CONNECTION ---
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

supabase: Optional[Client] = None

if url and key:
    try:
        supabase = create_client(url, key)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}", file=sys.stderr)
else:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in environment variables.", file=sys.stderr)

# --- 2. DATA MODELS (SCHEMAS) ---
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# --- 3. DATABASE FUNCTIONS ---

def sign_up_user(user: UserCreate):
    """Creates a new user in Supabase Auth"""
    if not supabase:
        raise Exception("Supabase client is not initialized")
    try:
        response = supabase.auth.sign_up({
            "email": user.email, 
            "password": user.password,
            "options": { "data": { "name": user.name } }
        })
        return response
    except Exception as e:
        raise e

def sign_in_user(user: UserLogin):
    """Logs in a user and returns the session"""
    if not supabase:
        raise Exception("Supabase client is not initialized")
    try:
        response = supabase.auth.sign_in_with_password({
            "email": user.email, 
            "password": user.password
        })
        return response
    except Exception as e:
        raise e

def save_booking(user_id: str, booking_type: str, details: Dict[str, Any], confirmation_id: str):
    """Inserts a confirmed booking into the 'bookings' table"""
    if not supabase:
        raise Exception("Supabase client is not initialized")
    try:
        data = {
            "user_id": user_id,
            "type": booking_type,
            "details": details,
            "confirmation_id": confirmation_id,
            "status": "confirmed"
        }
        return supabase.table("bookings").insert(data).execute()
    except Exception as e:
        print(f"Error saving booking: {e}")
        raise e

def get_user_bookings(user_id: str):
    """Fetches all bookings for a specific user"""
    if not supabase:
        return []
    try:
        response = supabase.table("bookings").select("*").eq("user_id", user_id).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching bookings: {e}")
        return []
