# image_caption.py
# Image-to-speech: caption an uploaded image (BLIP) and hand the caption to
# the existing TTS pipeline. This is the feature that turns the project from
# "text-to-speech toy" into an actual multimodal accessibility tool -- same
# spirit as the assistive scene-understanding project, applied here.

_captioner = None


def _get_captioner():
    global _captioner
    if _captioner is None:
        from transformers import pipeline
        _captioner = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-base"
        )
    return _captioner


def caption_image(image_path: str) -> str:
    """Return a natural-language caption for the image at image_path.
    Returns an error-prefixed string on failure instead of raising."""
    try:
        captioner = _get_captioner()
        result = captioner(image_path)
        return result[0]["generated_text"].strip()
    except Exception as e:
        return f"Error captioning image: {e}"
