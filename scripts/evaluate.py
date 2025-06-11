import torch
from datasets import load_dataset
from tqdm import tqdm
import yaml
import os
from PIL import Image
from io import BytesIO

# Import our custom model class
from models.model import PaliGemmaFineTune

def evaluate_winoground(model, processor, device, hf_token=None):
    """
    Evaluates a fine-tuned model on the Winoground benchmark using a loss-based method.
    """
    print("Loading Winoground test set for evaluation...")
    winoground_ds = load_dataset("facebook/winoground", split="test", token=hf_token)
    print(f"Winoground dataset loaded with {len(winoground_ds)} examples.")

    model.eval()
    model.to(device)

    image_correct = 0
    text_correct = 0
    group_correct = 0

    with torch.no_grad():
        for item in tqdm(winoground_ds, desc="Evaluating on Winoground"):
            # Extract images and captions
            image_0 = item['image_0']['bytes']
            image_1 = item['image_1']['bytes']
            caption_0 = item['caption_0']
            caption_1 = item['caption_1']

            # Function to calculate loss for a given image-caption pair
            def get_loss(image_bytes, caption_text):
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                
                # We provide the caption as both the prompt and the label (suffix)
                # to calculate the loss of the model generating that exact caption.
                inputs = processor(
                    text=caption_text, 
                    images=image, 
                    suffix=caption_text,
                    return_tensors="pt", 
                    padding="longest",
                    truncation=True,
                    max_length=128
                ).to(device)
                
                # The model's forward pass returns the loss directly
                outputs = model(**inputs)
                return outputs.loss.item()

            # Calculate loss for all four combinations
            loss_i0_c0 = get_loss(image_0, caption_0)
            loss_i0_c1 = get_loss(image_0, caption_1)
            loss_i1_c0 = get_loss(image_1, caption_0)
            loss_i1_c1 = get_loss(image_1, caption_1)

            # A lower loss means the model finds the pair more plausible.
            # Check Image Score: Does the model correctly match captions to their images?
            is_image_correct = (loss_i0_c0 < loss_i0_c1) and (loss_i1_c1 < loss_i1_c0)
            if is_image_correct:
                image_correct += 1
            
            # Check Text Score: Does the model correctly match images to their captions?
            is_text_correct = (loss_i0_c0 < loss_i1_c0) and (loss_i1_c1 < loss_i0_c1)
            if is_text_correct:
                text_correct += 1

            # Check Group Score: Is it correct on both axes?
            if is_image_correct and is_text_correct:
                group_correct += 1

    # Calculate final scores
    total_items = len(winoground_ds)
    image_score = image_correct / total_items
    text_score = text_correct / total_items
    group_score = group_correct / total_items
    
    return {
        "image_score": image_score,
        "text_score": text_score,
        "group_score": group_score
    }


def main():
    # --- Load Configuration ---
    try:
        with open('config/train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config/train_config.yaml not found. Please create it.")
        return
        
    hf_token = config.get('hf_access_token')
    
    # --- Load Fine-Tuned Model ---
    # The path should point to the directory where the Trainer saved the best model.
    # This is typically 'output_dir/checkpoint-XXXX'. We'll use the final saved model.
    model_path = os.path.join(config['output_dir'], "final_model")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}. Please run train.py first.")
        return
        
    print(f"Loading fine-tuned model from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use our custom `from_pretrained` class method
    model = PaliGemmaFineTune.from_pretrained(model_path, hf_access_token=hf_token)
    processor = model.processor

    # --- Perform Evaluation ---
    scores = evaluate_winoground(model, processor, device, hf_token)

    print("\n--- Winoground Evaluation Results ---")
    print(f"Image Score: {scores['image_score']:.4f}")
    print(f"Text Score:  {scores['text_score']:.4f}")
    print(f"Group Score: {scores['group_score']:.4f}")
    print("-------------------------------------")

if __name__ == "__main__":
    main()
