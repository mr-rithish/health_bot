"""
Simple WhatsApp webhook handler for Health Bot
Uses Rasa REST API
"""
import os
import logging
import asyncio
import aiohttp
from flask import Flask, request, Response
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Twilio client
twilio_client = Client(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# Rasa server URL
RASA_SERVER_URL = os.getenv("RASA_SERVER_URL", "http://localhost:5005/webhooks/rest/webhook")

async def send_message_to_rasa(message: str, sender: str):
    """Send message to Rasa and get response"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "sender": sender,
                "message": message
            }
            
            async with session.post(RASA_SERVER_URL, json=payload) as resp:
                if resp.status == 200:
                    responses = await resp.json()
                    return responses
                else:
                    logger.error(f"Rasa server error: {resp.status}")
                    return [{"text": "Sorry, I'm having trouble right now. Please try again later."}]
                    
    except Exception as e:
        logger.error(f"Error communicating with Rasa: {e}")
        return [{"text": "Sorry, I'm having trouble right now. Please try again later."}]

def send_whatsapp_message(to: str, message: str):
    """Send message via Twilio WhatsApp"""
    try:
        if not to.startswith("whatsapp:"):
            to = f"whatsapp:{to}"
        
        # Send actual message via Twilio
        twilio_message = twilio_client.messages.create(
            body=message,
            from_='whatsapp:+14155238886',  # Standard Twilio sandbox number
            to=to
        )
        logger.info(f"Sent WhatsApp message: {twilio_message.sid}")
        
    except Exception as e:
        logger.error(f"Error sending WhatsApp message: {e}")
        # Fallback to debug mode if sending fails
        logger.info(f"DEBUG - Would send to {to}: {message}")

@app.route("/", methods=["GET"])
def health_check():
    return {"status": "Health Bot WhatsApp Webhook is running!", "bot": "multilingual_health_assistant"}

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Handle incoming WhatsApp messages"""
    try:
        # Get message details
        sender = request.form.get('From', '')
        message_body = request.form.get('Body', '')
        
        logger.info(f"Received from {sender}: {message_body}")
        
        if sender and message_body:
            # Clean sender ID
            clean_sender = sender.replace("whatsapp:", "")
            
            # Send to Rasa and get responses
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            responses = loop.run_until_complete(
                send_message_to_rasa(message_body, clean_sender)
            )
            loop.close()
            
            # Send responses back via WhatsApp
            for response in responses:
                if "text" in response:
                    send_whatsapp_message(sender, response["text"])
        
        # Return empty TwiML response
        resp = MessagingResponse()
        return Response(str(resp), mimetype="text/xml")
        
    except Exception as e:
        logger.error(f"Error in webhook: {e}")
        resp = MessagingResponse()
        return Response(str(resp), mimetype="text/xml")

if __name__ == "__main__":
    print("🏥 Starting Health Bot WhatsApp Webhook...")
    print("📱 Make sure Rasa server is running on port 5005")
    app.run(host="0.0.0.0", port=5000, debug=False)
