from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import groq
import os
import json
import asyncio
import sys
from dotenv import load_dotenv
from typing import List, Dict

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
- If a user says something like “Book any ticket of Arijit Singh concert,” begin the concert ticket booking process with that artist.
- If the next message is “2,” understand that it likely means 2 tickets (especially if number of tickets hasn't been confirmed yet).
- For vague requests like “book any two tickets,” proceed using the most recent relevant booking details you have gathered so far.
- Before finalizing, confirm with the user: “Are you sure you want to confirm your booking with these details?”
- After confirmation, provide a fake booking confirmation number in the format: BOOK-XXXX-XXXX, where X is an alphanumeric character.

If at any point you don't have enough context (e.g., no type of booking or no event name), politely ask the user for the missing details.

Always be friendly, helpful, and concise in your responses.
"""

# Pydantic model for the user's input string
class UserInput(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: UserInput):
    """
    Endpoint to handle a single user query string and stream responses.
    Takes a JSON body like: {"query": "Your message here"}
    """

    async def stream_generator():
        try:
            # Prepare messages for Groq
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.query}
            ]

            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=True,
            )

            # Stream the response
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Format as SSE
                    yield f"data: {json.dumps({'type': 'text', 'value': content})}\n\n"
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)

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

