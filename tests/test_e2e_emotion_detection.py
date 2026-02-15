"""End-to-end tests for emotion_detection module.

These tests call the live Watson NLP API. They are marked with
`pytest.mark.e2e` so they can be skipped in CI or offline environments
with: pytest -m "not e2e"
"""

import json

import pytest
import requests

from emotion_detection import emotion_detector

WATSON_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"


def _api_is_reachable() -> bool:
    """Check if the Watson NLP API is reachable."""
    try:
        resp = requests.get(WATSON_URL, timeout=5)
        # Even a 405 means the server is up (GET not allowed on POST endpoint)
        return resp.status_code in (200, 405, 400, 404)
    except (requests.ConnectionError, requests.Timeout):
        return False


api_available = pytest.mark.skipif(
    not _api_is_reachable(),
    reason="Watson NLP API is not reachable"
)


@pytest.mark.e2e
class TestEmotionDetectorE2E:
    """End-to-end tests against the live Watson NLP API."""

    @api_available
    def test_live_api_returns_valid_json(self):
        result = emotion_detector("I love this amazing day")

        parsed = json.loads(result)
        assert "emotionPredictions" in parsed

    @api_available
    def test_live_api_returns_emotion_scores(self):
        result = emotion_detector("I am extremely happy and excited")

        parsed = json.loads(result)
        assert len(parsed["emotionPredictions"]) > 0
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert "joy" in emotions
        assert "anger" in emotions
        assert "sadness" in emotions
        assert "fear" in emotions
        assert "disgust" in emotions

    @api_available
    def test_live_joy_detection(self):
        result = emotion_detector("I am so happy and thrilled, this is wonderful!")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["joy"] > 0.5

    @api_available
    def test_live_anger_detection(self):
        result = emotion_detector("I am furious and outraged by this terrible situation!")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["anger"] > 0.3

    @api_available
    def test_live_sadness_detection(self):
        result = emotion_detector("I feel so sad and heartbroken, everything is lost")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        assert emotions["sadness"] > 0.3

    @api_available
    def test_live_emotion_scores_are_numeric(self):
        result = emotion_detector("This is a test sentence")

        parsed = json.loads(result)
        emotions = parsed["emotionPredictions"][0]["emotion"]
        for score in emotions.values():
            assert isinstance(score, (int, float))
            assert 0.0 <= score <= 1.0

    @api_available
    def test_live_response_is_string(self):
        result = emotion_detector("Testing the response type")

        assert isinstance(result, str)
        # Should be parseable as JSON
        json.loads(result)
