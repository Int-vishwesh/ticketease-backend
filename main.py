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
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta

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
SESSION_EXPIRY = timedelta(minutes=30)

# Pydantic model for the user's input
class UserInput(BaseModel):
    query: str
    session_id: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class Session:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.last_active = datetime.now()
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_active = datetime.now()
    
    def is_expired(self):
        return (datetime.now() - self.last_active) > SESSION_EXPIRY
    
    def get_messages(self):
        return self.messages

# Function to clean up expired sessions
def clean_expired_sessions():
    expired_sessions = [sid for sid, session in active_sessions.items() if session.is_expired()]
    for sid in expired_sessions:
        del active_sessions[sid]

@app.post("/chat")
async def chat(request: UserInput):
    """
    Endpoint to handle a single user query string and stream responses.
    Takes a JSON body like: {"query": "Your message here", "session_id": "optional-session-id"}
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

            # Stream the response preparation
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
            
            # Call Groq API
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=True,
            )

            # Collect the full response
            full_response = ""
            
            # Stream the response
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Format as SSE
                    yield f"data: {json.dumps({'type': 'text', 'value': content})}\n\n"
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
            
            # Add the assistant's response to the session
            session.add_message("assistant", full_response)
            
            # End of stream marker
            yield f"data: [DONE]\n\n"

        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            # Send error message via SSE
            yield f"data: {json.dumps({'type': 'error', 'value': error_message})}\n\n"
            yield f"data: [DONE]\n\n"

    # Return streaming response using the generator
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
