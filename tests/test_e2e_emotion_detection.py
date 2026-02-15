"""End-to-end tests for emotion_detection module.

These tests call the live Watson NLP API. They are marked with
`pytest.mark.e2e` so they can be skipped in CI or offline environments
with: pytest -m "not e2e"
"""

import pytest
import requests

from emotion_detection import emotion_detector, WATSON_URL

EXPECTED_KEYS = {"anger", "disgust", "fear", "joy", "sadness", "dominant_emotion"}


def _api_is_reachable() -> bool:
    try:
        resp = requests.get(WATSON_URL, timeout=5)
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
    def test_live_returns_dict(self):
        result = emotion_detector("I love this amazing day")
        assert isinstance(result, dict)

    @api_available
    def test_live_contains_all_keys(self):
        result = emotion_detector("I am extremely happy and excited")
        assert set(result.keys()) == EXPECTED_KEYS

    @api_available
    def test_live_joy_is_dominant(self):
        result = emotion_detector("I am so happy and thrilled, this is wonderful!")
        assert result["dominant_emotion"] == "joy"
        assert result["joy"] > 0.5

    @api_available
    def test_live_anger_detected(self):
        result = emotion_detector("I am furious and outraged by this terrible situation!")
        assert result["anger"] > 0.3

    @api_available
    def test_live_sadness_detected(self):
        result = emotion_detector("I feel so sad and heartbroken, everything is lost")
        assert result["sadness"] > 0.3

    @api_available
    def test_live_scores_are_numeric_and_bounded(self):
        result = emotion_detector("This is a test sentence")
        for key in ("anger", "disgust", "fear", "joy", "sadness"):
            assert isinstance(result[key], (int, float))
            assert 0.0 <= result[key] <= 1.0

    @api_available
    def test_live_dominant_is_valid_emotion(self):
        result = emotion_detector("Testing dominant emotion")
        assert result["dominant_emotion"] in ("anger", "disgust", "fear", "joy", "sadness")
