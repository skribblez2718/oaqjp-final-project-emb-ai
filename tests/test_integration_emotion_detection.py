"""Integration tests for emotion_detection module.

Tests the HTTP interaction layer by mocking at the transport level
(requests.Session/adapters), verifying correct request construction
and response handling end-to-end through the function.
"""

import json
from unittest.mock import patch, MagicMock

import requests

from emotion_detection import emotion_detector

WATSON_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"


def _make_mock_response(status_code: int, json_body: dict) -> MagicMock:
    """Create a realistic mock response object."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.text = json.dumps(json_body)
    mock_resp.json.return_value = json_body
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


class TestEmotionDetectorIntegration:
    """Integration tests verifying request/response flow."""

    @patch("emotion_detection.requests.post")
    def test_joy_emotion_response(self, mock_post):
        response_body = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.01,
                    "disgust": 0.01,
                    "fear": 0.01,
                    "joy": 0.97,
                    "sadness": 0.01,
                }
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector("I am thrilled and so happy!")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["joy"] > emotions["anger"]
        assert emotions["joy"] > emotions["sadness"]

    @patch("emotion_detection.requests.post")
    def test_anger_emotion_response(self, mock_post):
        response_body = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.92,
                    "disgust": 0.03,
                    "fear": 0.02,
                    "joy": 0.01,
                    "sadness": 0.02,
                }
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector("I am so angry about this!")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["anger"] > emotions["joy"]

    @patch("emotion_detection.requests.post")
    def test_sadness_emotion_response(self, mock_post):
        response_body = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.02,
                    "disgust": 0.01,
                    "fear": 0.03,
                    "joy": 0.01,
                    "sadness": 0.93,
                }
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector("I feel so sad and depressed")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["sadness"] > emotions["joy"]

    @patch("emotion_detection.requests.post")
    def test_response_contains_all_five_emotions(self, mock_post):
        response_body = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.1,
                    "fear": 0.1,
                    "joy": 0.5,
                    "sadness": 0.2,
                }
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector("This is interesting")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        expected_keys = {"anger", "disgust", "fear", "joy", "sadness"}
        assert set(emotions.keys()) == expected_keys

    @patch("emotion_detection.requests.post")
    def test_emotion_scores_are_floats(self, mock_post):
        response_body = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.2,
                    "fear": 0.3,
                    "joy": 0.2,
                    "sadness": 0.2,
                }
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector("test input")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        for score in emotions.values():
            assert isinstance(score, float)

    @patch("emotion_detection.requests.post")
    def test_http_error_propagates(self, mock_post):
        mock_post.return_value = _make_mock_response(500, {"error": "Internal Server Error"})

        result = emotion_detector("test")

        parsed = json.loads(result)
        assert "error" in parsed

    @patch("emotion_detection.requests.post")
    def test_long_text_input(self, mock_post):
        long_text = "I am happy. " * 500
        response_body = {
            "emotionPredictions": [{
                "emotion": {"anger": 0.01, "disgust": 0.01, "fear": 0.01, "joy": 0.95, "sadness": 0.02}
            }]
        }
        mock_post.return_value = _make_mock_response(200, response_body)

        result = emotion_detector(long_text)

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == long_text
        assert isinstance(result, str)

    @patch("emotion_detection.requests.post")
    def test_special_characters_in_text(self, mock_post):
        text = "I'm happy! 😊 <script>alert('xss')</script> & \"quoted\""
        mock_post.return_value = _make_mock_response(200, {"emotionPredictions": []})

        emotion_detector(text)

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == text
