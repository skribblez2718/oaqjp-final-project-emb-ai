import json
import logging

import requests

logger = logging.getLogger(__name__)

WATSON_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
WATSON_HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
REQUEST_TIMEOUT = 10


def emotion_detector(text_to_analyse):
    input_json = {"raw_document": {"text": text_to_analyse}}
    logger.info("Sending request to Watson NLP API...")
    try:
        response = requests.post(
            WATSON_URL, json=input_json, headers=WATSON_HEADERS, timeout=REQUEST_TIMEOUT
        )
        logger.info("Response received: status=%d", response.status_code)
    except requests.ConnectionError:
        logger.error("Connection failed — Watson API is unreachable")
        raise
    except requests.Timeout:
        logger.error("Request timed out after %ds", REQUEST_TIMEOUT)
        raise

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_dict = json.loads(response.text)
    emotions = response_dict["emotionPredictions"][0]["emotion"]

    anger = emotions["anger"]
    disgust = emotions["disgust"]
    fear = emotions["fear"]
    joy = emotions["joy"]
    sadness = emotions["sadness"]

    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
