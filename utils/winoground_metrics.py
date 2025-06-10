import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO

def calculate_winoground_scores(model, processor, winoground_dataset, device, max_new_tokens=10):
    """
    Calculates Image Score, Text Score, and Group Score for the Winoground dataset.
    This function performs inference on the model for each Winoground item.

    Args:
        model: The fine-tuned PaliGemma model (PaliGemmaFineTuneModel instance).
        processor: The PaliGemmaProcessor.
        winoground_dataset: An instance of WinogroundEvaluationDataset.
        device: The device to run inference on ('cuda' or 'cpu').
        max_new_tokens (int): Max tokens to generate for the 'Yes'/'No' answer.

    Returns:
        dict: A dictionary containing 'image_score', 'text_score', and 'group_score'.
    """
    model.eval()
    model.to(device)

    image_score_count = 0
    text_score_count = 0
    group_score_count = 0
    total_items = len(winoground_dataset)

    # Get token IDs for "Yes" and "No"
    # Ensure these are single tokens and not split by the tokenizer
    yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)

    with torch.no_grad():
        for i in tqdm(range(total_items), desc="Evaluating Winoground"):
            item = winoground_dataset.dataset[i] # Access raw dataset item

            image_0 = Image.open(BytesIO(item['image_0']['bytes'])).convert("RGB")
            image_1 = Image.open(BytesIO(item['image_1']['bytes'])).convert("RGB")
            caption_0 = item['caption_0']
            caption_1 = item['caption_1']

            # --- Get scores for all 4 combinations ---
            # We need a score that indicates how well the image matches the caption.
            # Since PaliGemma is text-out, we can prompt it with "Does this caption describe the image? Answer:"
            # and get the logit for "Yes" vs "No".

            # Function to get "Yes" logit for a given image-text pair
            def get_yes_logit(img, text):
                prompt = f"Does this caption describe the image? Answer:"
                inputs = processor(images=img, text=prompt + text, return_tensors="pt", padding=True).to(device)
                
                # Generate a short answer (Yes/No)
                # We need to get the logits for the first generated token
                # The model's forward pass (from PaliGemmaFineTuneModel) returns CausalLMOutputWithPast
                # which contains logits. We need to ensure we get logits for the *first* token after the prompt.
                
                # To get the logit for 'Yes'/'No', we can generate one token and check its probability.
                # Or, more directly, we can pass the prompt and get the logits for the *next* token.
                
                # Let's use the model's generate method for simplicity, but configure it to
                # only generate one token and return scores.
                
                # The `PaliGemmaForConditionalGeneration`'s `generate` method can return `scores`.
                # We need to ensure our `PaliGemmaFineTuneModel` wrapper exposes this.
                # For now, let's assume we can get the logits for the first token after the prompt.
                
                # A more direct way without `generate` is to get the logits for the first token
                # after the prompt from the `forward` pass.
                
                # Let's adapt the forward pass to give us the logits for the *next* token after the prompt.
                # The `PaliGemmaFineTuneModel`'s forward returns `outputs.logits`.
                # The logits are for the *entire* sequence. We need the logits for the token *after* the prompt.
                
                # The `input_ids` passed to the model are `prompt_ids + text_ids`.
                # The logits are `(batch_size, seq_len, vocab_size)`.
                # We need the logits at `seq_len - 1` (the position *after* the last token of `prompt + text`).
                # This is tricky because the model is autoregressive.
                
                # Simpler approach: Use `model.model.generate` and check the log-likelihood of "Yes" vs "No".
                # This requires the base model to be accessible.
                
                # Let's modify the `PaliGemmaFineTuneModel` to have a `get_matching_score` method
                # that uses the base model's `generate` or `forward` to get a score for "Yes".
                
                # For now, let's simplify: we will generate a single token and check if it's "Yes" or "No".
                # This is a common heuristic for VQA-like tasks.
                
                # Prepare inputs for generation
                inputs_for_gen = processor(images=img, text=prompt + text, return_tensors="pt").to(device)
                
                # Generate one token
                generated_ids = model.model.generate(
                    **inputs_for_gen,
                    max_new_tokens=1,
                    do_sample=False, # Greedy decoding
                    return_dict_in_generate=True,
                    output_scores=True # Get scores for generated tokens
                )
                
                # Get the scores (logits) for the first generated token
                # scores contains logits for the first generated token across the vocabulary
                first_token_logits = generated_ids.scores.squeeze(0) # shape: (vocab_size,)
                
                # Get the logit for "Yes" and "No" tokens
                yes_logit = first_token_logits[yes_token_id].item()
                no_logit = first_token_logits[no_token_id].item()
                
                # Return the difference, or just the yes_logit, assuming higher is better for "Yes"
                return yes_logit - no_logit # Higher value means more likely to be "Yes"
            
            # Calculate scores for all four pairs
            score_i0c0 = get_yes_logit(image_0, caption_0)
            score_i0c1 = get_yes_logit(image_0, caption_1)
            score_i1c0 = get_yes_logit(image_1, caption_0)
            score_i1c1 = get_yes_logit(image_1, caption_1)

            # Winoground Image Score:
            # For I0, C0 must be preferred over C1. For I1, C1 must be preferred over C0.
            if (score_i0c0 > score_i0c1) and (score_i1c1 > score_i1c0):
                image_score_count += 1

            # Winoground Text Score:
            # For C0, I0 must be preferred over I1. For C1, I1 must be preferred over I0.
            if (score_i0c0 > score_i1c0) and (score_i1c1 > score_i0c1):
                text_score_count += 1

            # Winoground Group Score:
            # Both Image Score and Text Score conditions must be met.
            if ((score_i0c0 > score_i0c1) and (score_i1c1 > score_i1c0)) and \
               ((score_i0c0 > score_i1c0) and (score_i1c1 > score_i0c1)):
                group_score_count += 1

    image_score = image_score_count / total_items
    text_score = text_score_count / total_items
    group_score = group_score_count / total_items

    return {
        "winoground_image_score": image_score,
        "winoground_text_score": text_score,
        "winoground_group_score": group_score,
    }

if __name__ == '__main__':
    # Dummy Model and Processor for testing
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from datasets import Dataset as HFDataset
    import numpy as np
    import pandas as pd

    class DummyPaliGemmaModel(nn.Module):
        def __init__(self, vocab_size=257216, hidden_size=2048):
            super().__init__()
            # Simulate a very simple model that returns random logits
            self.dummy_logits = torch.randn(1, vocab_size) # For single token generation
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size

        def generate(self, **kwargs):
            # Simulate generate method returning scores for a single token
            # This is a highly simplified mock
            dummy_scores = [torch.randn(1, self.vocab_size)] # Logits for the first token
            
            # Mock the generate output structure
            class MockGenerateOutput:
                def __init__(self, scores):
                    self.scores = scores
                    self.sequences = torch.tensor([]) # Dummy sequence

            return MockGenerateOutput(dummy_scores)

    class DummyProcessor:
        def __init__(self):
            # Mock tokenizer with encode method
            class MockTokenizer:
                def encode(self, text, add_special_tokens=False):
                    if text == "Yes": return 
                    if text == "No": return 
                    return [random.randint(0, 257215) for _ in range(len(text.split()))]
            self.tokenizer = MockTokenizer()
        
        def __call__(self, images, text, return_tensors="pt", padding=True, truncation=True, max_length=None):
            # Mock processor call
            batch_size = len(text) if isinstance(text, list) else 1
            return {
                "pixel_values": torch.randn(batch_size, 3, 224, 224),
                "input_ids": torch.randint(0, 257216, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10)
            }

    # Create dummy Winoground-like dataset
    dummy_data = {
        'image_0': [{'bytes': b'dummy_img_0'} for _ in range(2)],
        'image_1': [{'bytes': b'dummy_img_1'} for _ in range(2)],
        'caption_0': ["a cat on a mat", "a dog in a car"],
        'caption_1': ["a mat on a cat", "a car in a dog"],
    }
    dummy_hf_dataset = HFDataset.from_dict(dummy_data)

    class DummyWinogroundEvaluationDataset:
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)

    dummy_model_instance = DummyPaliGemmaModel()
    dummy_processor_instance = DummyProcessor()
    dummy_winoground_eval_dataset = DummyWinogroundEvaluationDataset(dummy_hf_dataset)

    # Mock PIL Image.open
    def mock_image_open(bytes_io):
        class MockImage:
            def convert(self, mode): return self
        return MockImage()
    
    Image.open = mock_image_open

    print("Running dummy Winoground evaluation...")
    scores = calculate_winoground_scores(
        model=dummy_model_instance,
        processor=dummy_processor_instance,
        winoground_dataset=dummy_winoground_eval_dataset,
        device='cpu'
    )
    print(f"Dummy Winoground Scores: {scores}")