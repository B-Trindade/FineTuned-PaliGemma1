import os
import pytest
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# Define the model name to test
MODEL_NAME = "google/paligemma-3b-mix-224"

@pytest.fixture(scope="session")
def hf_access_token():
    """
    Fixture to retrieve the Hugging Face access token from environment variables.
    This ensures the token is not hardcoded and promotes secure practices.
    """
    token = os.environ.get("HF_ACCESS_TOKEN")
    if token is None:
        pytest.skip(
            "HF_ACCESS_TOKEN environment variable not set. "
            "Please set it to run Hugging Face connection tests."
        )
    return token

def test_can_load_paligemma_processor(hf_access_token):
    """
    Tests if the PaliGemmaProcessor can be successfully loaded from Hugging Face Hub.
    This implicitly checks network connectivity and token validity for the processor.
    """
    print(f"\nAttempting to load {MODEL_NAME} processor...")
    try:
        processor = PaliGemmaProcessor.from_pretrained(MODEL_NAME, token=hf_access_token)
        assert processor is not None, "PaliGemmaProcessor did not load successfully."
        print(f"Successfully loaded {MODEL_NAME} processor.")
    except Exception as e:
        pytest.fail(f"Failed to load {MODEL_NAME} processor: {e}")

def test_can_load_paligemma_model(hf_access_token):
    """
    Tests if the PaliGemmaForConditionalGeneration model can be successfully loaded
    from Hugging Face Hub. This checks network connectivity, token validity, and
    access permissions for the model weights.
    """
    print(f"\nAttempting to load {MODEL_NAME} model...")
    try:
        # Using a small torch_dtype like torch.float16 or torch.bfloat16
        # can reduce memory usage during the test, especially for large models.
        # Ensure your environment supports bfloat16 (newer GPUs, or CPU fallback).
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
            token=hf_access_token
        )
        assert model is not None, "PaliGemmaForConditionalGeneration did not load successfully."
        print(f"Successfully loaded {MODEL_NAME} model.")
    except Exception as e:
        # Catch specific errors for better diagnostics if needed
        # For example, OSError for file not found, requests.exceptions.HTTPError for network errors
        pytest.fail(f"Failed to load {MODEL_NAME} model. Ensure your HF_ACCESS_TOKEN is valid, "
                     "and you have accepted the model's terms on Hugging Face Hub: "
                     f"{MODEL_NAME}\nError: {e}")

# Note: You can add more specific tests here, e.g., to check if model.config is loaded.
