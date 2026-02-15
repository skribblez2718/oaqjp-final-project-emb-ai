"""Unit tests for emotion_detection module.

Tests the emotion_detector function in isolation by mocking requests.post.
"""

import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from emotion_detection import emotion_detector, WATSON_URL, WATSON_HEADERS, REQUEST_TIMEOUT


class TestEmotionDetectorUnit:
    """Unit tests with requests.post fully mocked."""

    @patch("emotion_detection.requests.post")
    def test_returns_response_text(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = '{"emotionPredictions": [{"emotion": {"joy": 0.95}}]}'
        mock_post.return_value = mock_response

        result = emotion_detector("I am so happy today")

        assert result == mock_response.text

    @patch("emotion_detection.requests.post")
    def test_calls_correct_url(self, mock_post):
        mock_post.return_value = MagicMock(text="{}")

        emotion_detector("test text")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == WATSON_URL

    @patch("emotion_detection.requests.post")
    def test_sends_correct_headers(self, mock_post):
        mock_post.return_value = MagicMock(text="{}")

        emotion_detector("test text")

        call_args = mock_post.call_args
        assert call_args[1]["headers"] == WATSON_HEADERS

    @patch("emotion_detection.requests.post")
    def test_sends_correct_json_payload(self, mock_post):
        mock_post.return_value = MagicMock(text="{}")

        emotion_detector("I love this product")

        call_args = mock_post.call_args
        expected_json = {"raw_document": {"text": "I love this product"}}
        assert call_args[1]["json"] == expected_json

    @patch("emotion_detection.requests.post")
    def test_sends_timeout(self, mock_post):
        mock_post.return_value = MagicMock(text="{}")

        emotion_detector("test text")

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == REQUEST_TIMEOUT

    @patch("emotion_detection.requests.post")
    def test_passes_text_argument_in_payload(self, mock_post):
        mock_post.return_value = MagicMock(text="{}")

        emotion_detector("different text here")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == "different text here"

    @patch("emotion_detection.requests.post")
    def test_returns_string_type(self, mock_post):
        mock_post.return_value = MagicMock(text='{"result": "ok"}')

        result = emotion_detector("test")

        assert isinstance(result, str)

    @patch("emotion_detection.requests.post")
    def test_empty_text_input(self, mock_post):
        mock_post.return_value = MagicMock(text='{"emotionPredictions": []}')

        result = emotion_detector("")

        call_args = mock_post.call_args
        assert call_args[1]["json"]["raw_document"]["text"] == ""
        assert isinstance(result, str)

    @patch("emotion_detection.requests.post")
    def test_response_is_valid_json_string(self, mock_post):
        expected = json.dumps({"emotionPredictions": [{"emotion": {"anger": 0.8}}]})
        mock_post.return_value = MagicMock(text=expected)

        result = emotion_detector("I am furious")

        parsed = json.loads(result)
        assert "emotionPredictions" in parsed

    @patch("emotion_detection.requests.post")
    def test_connection_error_raises(self, mock_post):
        mock_post.side_effect = requests.ConnectionError("Connection refused")

        with pytest.raises(requests.ConnectionError):
            emotion_detector("test")

    @patch("emotion_detection.requests.post")
    def test_timeout_error_raises(self, mock_post):
        mock_post.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(requests.Timeout):
            emotion_detector("test")
