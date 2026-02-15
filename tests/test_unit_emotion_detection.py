"""Unit tests for emotion_detection module.

Tests the emotion_detector function in isolation by mocking requests.post.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from EmotionDetection.emotion_detection import emotion_detector, WATSON_URL, WATSON_HEADERS, REQUEST_TIMEOUT

SAMPLE_API_RESPONSE = {
    "emotionPredictions": [{
        "emotion": {
            "anger": 0.01,
            "disgust": 0.02,
            "fear": 0.03,
            "joy": 0.91,
            "sadness": 0.03,
        }
    }]
}

EXPECTED_KEYS = {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"}


def _mock_response(body: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(body)
    mock_resp.status_code = 200
    return mock_resp


class TestEmotionDetectorRequest:
    """Tests verifying the outgoing HTTP request is constructed correctly."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_calls_correct_url(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)
        emotion_detector("test text")

        call_args = mock_post.call_args
        assert call_args[0][0] == WATSON_URL

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sends_correct_headers(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)
        emotion_detector("test text")

        call_args = mock_post.call_args
        assert call_args[1]["headers"] == WATSON_HEADERS

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sends_correct_json_payload(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)
        emotion_detector("I love this product")

        call_args = mock_post.call_args
        assert call_args[1]["json"] == {"raw_document": {"text": "I love this product"}}

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sends_timeout(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)
        emotion_detector("test text")

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == REQUEST_TIMEOUT

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_passes_text_argument_in_payload(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)
        emotion_detector("different text here")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == "different text here"


class TestEmotionDetectorReturnFormat:
    """Tests verifying the returned dict has the correct shape and content."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_returns_dict(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)

        result = emotion_detector("I am happy")

        assert isinstance(result, dict)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_contains_all_expected_keys(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)

        result = emotion_detector("I am happy")

        assert set(result.keys()) == EXPECTED_KEYS

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_scores_are_floats(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)

        result = emotion_detector("I am happy")

        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            assert isinstance(result[key], float)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_scores_match_api_response(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)

        result = emotion_detector("I am happy")

        assert result["anger"] == 0.01
        assert result["disgust"] == 0.02
        assert result["fear"] == 0.03
        assert result["joy"] == 0.91
        assert result["sadness"] == 0.03

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_emotion_is_string(self, mock_post):
        mock_post.return_value = _mock_response(SAMPLE_API_RESPONSE)

        result = emotion_detector("I am happy")

        assert isinstance(result["dominant_emotion"], str)


class TestDominantEmotion:
    """Tests verifying the dominant emotion logic."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_is_joy(self, mock_post):
        body = {"emotionPredictions": [{"emotion": {
            "anger": 0.01, "disgust": 0.01, "fear": 0.01, "joy": 0.95, "sadness": 0.02
        }}]}
        mock_post.return_value = _mock_response(body)

        result = emotion_detector("I am so happy")
        assert result["dominant_emotion"] == "joy"

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_is_anger(self, mock_post):
        body = {"emotionPredictions": [{"emotion": {
            "anger": 0.88, "disgust": 0.05, "fear": 0.02, "joy": 0.01, "sadness": 0.04
        }}]}
        mock_post.return_value = _mock_response(body)

        result = emotion_detector("I am furious")
        assert result["dominant_emotion"] == "anger"

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_is_sadness(self, mock_post):
        body = {"emotionPredictions": [{"emotion": {
            "anger": 0.02, "disgust": 0.01, "fear": 0.03, "joy": 0.01, "sadness": 0.93
        }}]}
        mock_post.return_value = _mock_response(body)

        result = emotion_detector("I feel so sad")
        assert result["dominant_emotion"] == "sadness"

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_is_fear(self, mock_post):
        body = {"emotionPredictions": [{"emotion": {
            "anger": 0.05, "disgust": 0.02, "fear": 0.85, "joy": 0.03, "sadness": 0.05
        }}]}
        mock_post.return_value = _mock_response(body)

        result = emotion_detector("I am terrified")
        assert result["dominant_emotion"] == "fear"

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_dominant_is_disgust(self, mock_post):
        body = {"emotionPredictions": [{"emotion": {
            "anger": 0.05, "disgust": 0.80, "fear": 0.05, "joy": 0.05, "sadness": 0.05
        }}]}
        mock_post.return_value = _mock_response(body)

        result = emotion_detector("That is disgusting")
        assert result["dominant_emotion"] == "disgust"


class TestEmotionDetectorErrors:
    """Tests verifying error handling."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_connection_error_raises(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            emotion_detector("test")

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_timeout_error_raises(self, mock_post):
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            emotion_detector("test")
