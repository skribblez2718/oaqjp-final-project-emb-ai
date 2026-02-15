import logging

from EmotionDetection import emotion_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def main():
    result = emotion_detector("I love this new technology.")
    print(result)


if __name__ == "__main__":
    main()
