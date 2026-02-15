"""Integration tests for emotion_detection module.

Tests the full request/response flow by mocking at the HTTP layer,
verifying correct parsing, emotion extraction, and dominant emotion logic.
"""

import json
from unittest.mock import patch, MagicMock

import requests

from EmotionDetection.emotion_detection import emotion_detector, WATSON_URL

EXPECTED_KEYS = {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"}


def _make_mock_response(status_code: int, json_body: dict) -> MagicMock:
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.text = json.dumps(json_body)
    mock_resp.json.return_value = json_body
    mock_resp.headers = {"Content-Type": "application/json"}
    return mock_resp


def _api_body(anger=0.1, disgust=0.1, fear=0.1, joy=0.5, sadness=0.2):
    return {"emotionPredictions": [{"emotion": {
        "anger": anger, "disgust": disgust, "fear": fear, "joy": joy, "sadness": sadness
    }}]}


class TestEmotionDetectorIntegration:
    """Integration tests verifying full request→parse→extract→return flow."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_dominant_response(self, mock_post):
        mock_post.return_value = _make_mock_response(200, _api_body(joy=0.97, anger=0.01))

        result = emotion_detector("I am thrilled and so happy!")

        assert result["dominant_emotion"] == "joy"
        assert result["joy"] == 0.97
        assert result["anger"] == 0.01

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_dominant_response(self, mock_post):
        mock_post.return_value = _make_mock_response(200, _api_body(anger=0.92, joy=0.01))

        result = emotion_detector("I am so angry about this!")

        assert result["dominant_emotion"] == "anger"
        assert result["anger"] == 0.92

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_dominant_response(self, mock_post):
        mock_post.return_value = _make_mock_response(200, _api_body(sadness=0.93, joy=0.01))

        result = emotion_detector("I feel so sad and depressed")

        assert result["dominant_emotion"] == "sadness"
        assert result["sadness"] == 0.93

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_response_contains_all_six_keys(self, mock_post):
        mock_post.return_value = _make_mock_response(200, _api_body())

        result = emotion_detector("This is interesting")

        assert set(result.keys()) == EXPECTED_KEYS

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_scores_are_floats(self, mock_post):
        mock_post.return_value = _make_mock_response(200, _api_body())

        result = emotion_detector("test input")

        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            assert isinstance(result[key], float)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_long_text_input(self, mock_post):
        long_text = "I am happy. " * 500
        mock_post.return_value = _make_mock_response(200, _api_body(joy=0.95))

        result = emotion_detector(long_text)

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == long_text
        assert result["dominant_emotion"] == "joy"

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_special_characters_in_text(self, mock_post):
        text = "I'm happy! 😊 <script>alert('xss')</script> & \"quoted\""
        mock_post.return_value = _make_mock_response(200, _api_body(joy=0.8))

        result = emotion_detector(text)

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == text
        assert isinstance(result, dict)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_close_scores_picks_highest(self, mock_post):
        mock_post.return_value = _make_mock_response(
            200, _api_body(anger=0.30, disgust=0.31, fear=0.29, joy=0.05, sadness=0.05)
        )

        result = emotion_detector("mixed feelings")

        assert result["dominant_emotion"] == "disgust"
