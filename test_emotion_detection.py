import json
import unittest
from unittest.mock import patch, MagicMock

from EmotionDetection.emotion_detection import emotion_detector

# Simulated API responses for each test statement
API_RESPONSES = {
    "I am glad this happened": {
        "emotionPredictions": [{"emotion": {
            "anger": 0.01, "disgust": 0.01, "fear": 0.01, "joy": 0.95, "sadness": 0.02
        }}]
    },
    "I am really mad about this": {
        "emotionPredictions": [{"emotion": {
            "anger": 0.92, "disgust": 0.03, "fear": 0.01, "joy": 0.01, "sadness": 0.03
        }}]
    },
    "I feel disgusted just hearing about this": {
        "emotionPredictions": [{"emotion": {
            "anger": 0.05, "disgust": 0.88, "fear": 0.02, "joy": 0.01, "sadness": 0.04
        }}]
    },
    "I am so sad about this": {
        "emotionPredictions": [{"emotion": {
            "anger": 0.02, "disgust": 0.01, "fear": 0.03, "joy": 0.01, "sadness": 0.93
        }}]
    },
    "I am really afraid that this will happen": {
        "emotionPredictions": [{"emotion": {
            "anger": 0.03, "disgust": 0.01, "fear": 0.90, "joy": 0.02, "sadness": 0.04
        }}]
    },
}


def _mock_post(url, json=None, headers=None, timeout=None):
    """Return a mock response matching the input text."""
    text = json["raw_document"]["text"]
    mock_resp = MagicMock()
    mock_resp.text = __import__("json").dumps(API_RESPONSES[text])
    mock_resp.status_code = 200
    return mock_resp


class TestEmotionDetection(unittest.TestCase):
    """Unit tests for the EmotionDetection package."""

    @patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_post)
    def test_joy(self, mock_post):
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_post)
    def test_anger(self, mock_post):
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_post)
    def test_disgust(self, mock_post):
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    @patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_post)
    def test_sadness(self, mock_post):
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("EmotionDetection.emotion_detection.requests.post", side_effect=_mock_post)
    def test_fear(self, mock_post):
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
