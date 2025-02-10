from django.shortcuts import render
from openai import AzureOpenAI
import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view

# OpenAI API Config
client = AzureOpenAI(api_key=settings.AZURE_OPENAI_API_KEY, azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, api_version="2024-07-01-preview",)
# Azure AI Text Analytics Headers
TEXT_ANALYTICS_HEADERS = {
    "Ocp-Apim-Subscription-Key": settings.AZURE_TEXT_ANALYTICS_KEY,
    "Content-Type": "application/json"
}

@api_view(["POST"])
def chatbot_response(request):
    user_input = request.data.get("message", "")

    if not user_input:
        return Response({"error": "No input received"}, status=400)

    # Analyze sentiment using Azure Text Analytics
    sentiment_result = analyze_sentiment(user_input)

    # Generate AI response
    bot_response = generate_ai_response(user_input)

    # Detect distress and provide emergency response
    if sentiment_result and sentiment_result.get("sentiment") == "negative":
        bot_response += "\n\nIf you're in distress, please consider reaching out to a mental health professional or helpline on https://www.whatseatingmymind.com/emergency-hotline-numbers"

    return Response({"response": bot_response})


def analyze_sentiment(text):
    """Analyze sentiment using Azure AI Text Analytics"""
    url = f"{settings.AZURE_TEXT_ANALYTICS_ENDPOINT}/text/analytics/v3.1/sentiment"
    data = {"documents": [{"id": "1", "language": "en", "text": text}]}
    response = requests.post(url, headers=TEXT_ANALYTICS_HEADERS, json=data)

    if response.status_code == 200:
        sentiment_data = response.json()
        return sentiment_data["documents"][0]
    return None

def generate_ai_response(user_input):
    """Generate response using Azure OpenAI GPT-4 Turbo"""
    response = client.chat.completions.create(model="gpt-4o",
    messages=[{"role": "user", "content": user_input}])
    return response.choices[0].message.content

