print("Loading main.py clean version...")
from fastapi import FastAPI, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import groq
import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
import db  # Import the new database module

# Load environment variables
load_dotenv()

app = FastAPI(title="Ticket Booking AI Backend")

# Configure CORS - Allow all origins in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_api_key = os.getenv("GROQ_API_KEY", "")
if not groq_api_key:
    print("Error: GROQ_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)
groq_client = groq.Groq(api_key=groq_api_key)

# System prompt for the AI
SYSTEM_PROMPT = """
You are a helpful ticket booking assistant that helps users book tickets for various events and appointments.
You can handle bookings for:
1. Doctor appointments
2. Amusement park tickets
3. Movie tickets
4. Concert tickets
5. Sports events
6. And other similar bookings

Your job is to collect booking information gradually and naturally in a conversation, even if the user provides it across multiple messages.

Instructions:
- Keep track of the conversation context. Assume the user is continuing from the last message unless they clearly start a new request.
- If a user says something like "Book any ticket of Arijit Singh concert," begin the concert ticket booking process with that artist.
- If the next message is "2," understand that it likely means 2 tickets (especially if number of tickets hasn't been confirmed yet with previous message contexts).
- For vague requests like "book any two tickets," proceed using the most recent relevant booking details you have gathered so far.
- Before finalizing, confirm with the user: "Are you sure you want to confirm your booking with these details?"
- After confirmation, provide a fake booking confirmation number in the format: BOOK-XXXX-XXXX, where X is an alphanumeric character.

If at any point you don't have enough context (e.g., no type of booking or no event name), politely ask the user for the missing details.

Always be friendly, helpful, and concise in your responses.
"""

# In-memory session storage 
# In a production environment, you should use Redis or a database
active_sessions = {}

# Session expiry time (30 minutes)
# Session expiry time (30 minutes)
SESSION_EXPIRY = timedelta(minutes=30)

class Session:
    def __init__(self):
        self.messages = []
        self.last_activity = datetime.now()

    def add_message(self, role, content):
        self.last_activity = datetime.now()
        # Ensure we don't duplicate tool calls or results if they are already handled logic is complex, 
        # but for now just appending is standard.
        # Note: 'content' can be None for assistant messages with tool calls
        if content or role == "tool" or (role == "assistant" and content is None):
             self.messages.append({"role": role, "content": content})

    def get_messages(self):
        self.last_activity = datetime.now()
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages

def clean_expired_sessions():
    """Remove sessions that have been inactive for longer than SESSION_EXPIRY"""
    now = datetime.now()
    expired_sessions = [
        sid for sid, session in active_sessions.items() 
        if now - session.last_activity > SESSION_EXPIRY
    ]
    for sid in expired_sessions:
        del active_sessions[sid]

# Pydantic model for the user's input
class UserInput(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

# Tool definition for Groq
tools = [
    {
        "type": "function",
        "function": {
            "name": "save_booking",
            "description": "Save a confirmed booking to the database. Call this ONLY after the user has explicitly confirmed the booking details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_type": {
                        "type": "string",
                        "description": "The type of booking (e.g., 'movie', 'concert', 'doctor', 'amusement_park', 'sports')",
                    },
                    "details": {
                        "type": "object",
                        "description": "A dictionary containing all booking details (e.g., artist, venue, time, seats, doctor name, etc.)",
                    },
                    "confirmation_id": {
                        "type": "string",
                        "description": "The generated confirmation ID (e.g., BOOK-XXXX-XXXX)",
                    }
                },
                "required": ["booking_type", "details", "confirmation_id"],
            },
        },
    }
]
@app.get("/")
async def root():
    return {"message": "TicketEase Backend running!"}

@app.post("/chat")
async def chat(request: UserInput):
    """
    Endpoint to handle a single user query string and stream responses.
    Takes a JSON body like: {"query": "Your message here", "session_id": "optional-session-id", "user_id": "optional-user-id"}
    Returns a streaming response with the AI's reply and a session ID.
    """
    # Clean expired sessions first
    clean_expired_sessions()
    
    # Get or create session
    session_id = request.session_id
    if not session_id or session_id not in active_sessions:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = Session()
    
    session = active_sessions[session_id]
    
    # Add user message to session
    session.add_message("user", request.query)

    async def stream_generator():
        try:
            # Get all messages from the session
            messages = session.get_messages()

            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # Prepare args for Groq
            api_args = {
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stream": True,
            }

            # Only add tools if user_id is present (we need user_id to save)
            if request.user_id:
                api_args["tools"] = tools
                api_args["tool_choice"] = "auto"
            
            # Call Groq API
            completion = groq_client.chat.completions.create(**api_args)

            full_response = ""
            tool_calls = []
            current_tool_call = None

            for chunk in completion:
                delta = chunk.choices[0].delta if chunk.choices else None
                
                # Check for tool_calls
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id: # New tool call
                            if current_tool_call:
                                tool_calls.append(current_tool_call)
                            current_tool_call = {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments or ""
                                },
                                "type": tc.type
                            }
                        elif current_tool_call: # Append arguments
                            current_tool_call["function"]["arguments"] += (tc.function.arguments or "")

                # Check for content
                if delta and delta.content:
                    content = delta.content
                    full_response += content
                    yield f"data: {json.dumps({'type': 'text', 'value': content})}\n\n"
                    await asyncio.sleep(0.01)
            
            # Handle any completed tool calls
            if current_tool_call:
                tool_calls.append(current_tool_call)

            if tool_calls:
                # Add the assistant's message with tool calls to history
                session.messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "content": full_response or None # Content might be empty if only tool called
                })

                # Execute tools
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                    
                    if func_name == "save_booking":
                        try:
                            args = json.loads(args_str)
                            # Call the db function
                            db.save_booking(
                                user_id=request.user_id,
                                booking_type=args.get("booking_type"),
                                details=args.get("details"),
                                confirmation_id=args.get("confirmation_id")
                            )
                            tool_result = json.dumps({"status": "success", "message": "Booking saved to database."})
                        except Exception as e:
                            print(f"Error saving booking: {e}")
                            tool_result = json.dumps({"status": "error", "message": str(e)})
                        
                        # Add tool result to messages
                        session.messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result
                        })

                # Call LLM again to generate final response (confirmation to user)
                completion_final = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=session.get_messages(),
                    stream=True
                )
                
                # Reset full_response for the final message
                full_response = ""

                # Stream the final response
                for chunk in completion_final:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'type': 'text', 'value': content})}\n\n"
                        await asyncio.sleep(0.01)

                # Add the final text response to history
                session.add_message("assistant", full_response)
                
            else:
                # No tools called, just save the response
                session.add_message("assistant", full_response)
            
            yield f"data: [DONE]\n\n"

        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'value': error_message})}\n\n"
            yield f"data: [DONE]\n\n"

    # Return streaming response using the generator
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# --- New Endpoints for Authentication & Data ---

@app.post("/signup")
async def signup(user: db.UserCreate):
    """Endpoint to register a new user"""
    try:
        response = db.sign_up_user(user)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(user: db.UserLogin):
    """Endpoint to login a user"""
    try:
        response = db.sign_in_user(user)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/bookings/{user_id}")
async def get_bookings(user_id: str):
    """Endpoint to get user bookings"""
    return db.get_user_bookings(user_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

class BookingCreate(BaseModel):
    user_id: str
    booking_type: str
    details: Dict[str, Any]
    confirmation_id: str

@app.post("/bookings")
async def create_booking(booking: BookingCreate):
    """Endpoint to create a new booking directly"""
    try:
        response = db.save_booking(
            user_id=booking.user_id,
            booking_type=booking.booking_type,
            details=booking.details,
            confirmation_id=booking.confirmation_id
        )
        return {"status": "success", "message": "Booking saved to database."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


#-------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
