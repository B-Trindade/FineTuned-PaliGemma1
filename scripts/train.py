import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import random
from PIL import Image
import numpy as np

# Import our custom components
from models.pali_gemma_ft import PaliGemmaFineTuneModel
from utils.losses import TripletLoss

# --- Dummy Winoground-like Dataset for Demonstration ---
# In a real scenario, this would load the actual Winoground dataset
# and create triplets (image_0, caption_0_good, caption_0_bad) or similar.
class DummyWinogroundDataset(Dataset):
    def __init__(self, processor, num_samples=100):
        self.processor = processor
        self.num_samples = num_samples
        self.data = []

        # Create dummy data: (image_path, positive_caption, negative_caption)
        # For Winoground, positive_caption is the correct one for the image.
        # Negative_caption is lexically similar but semantically incorrect for the image.
        for i in range(num_samples):
            # Dummy image path (we'll generate dummy images on the fly)
            image_id = f"image_{i:03d}.png"
            positive_caption = f"The {['red', 'blue', 'green'][i % 3]} dog is playing with the {['ball', 'frisbee'][i % 2]}."
            negative_caption = f"The {['blue', 'green', 'red'][i % 3]} dog is playing with the {['frisbee', 'ball'][i % 2]}." # Lexically similar but wrong color/object
            self.data.append((image_id, positive_caption, negative_caption))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # In a real dataset, you'd load the actual image from disk.
        # Here, we generate a dummy image.
        # A simple black square image
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        _, positive_caption, negative_caption = self.data[idx]

        # Process inputs. Note: We'll process images and captions separately
        # to form anchor-positive-negative triplets for the model.
        # The model's forward pass expects pixel_values, input_ids, attention_mask for each.

        # Anchor (image-positive_caption pair)
        anchor_inputs = self.processor(images=dummy_image, text=positive_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        # Positive (image-positive_caption pair, which is effectively the same as anchor, but explicit for triplet)
        positive_inputs = self.processor(images=dummy_image, text=positive_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        # Negative (image-negative_caption pair)
        negative_inputs = self.processor(images=dummy_image, text=negative_caption, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        return {
            "anchor_pixel_values": anchor_inputs["pixel_values"].squeeze(0),
            "anchor_input_ids": anchor_inputs["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_inputs["attention_mask"].squeeze(0),
            "positive_pixel_values": positive_inputs["pixel_values"].squeeze(0),
            "positive_input_ids": positive_inputs["input_ids"].squeeze(0),
            "positive_attention_mask": positive_inputs["attention_mask"].squeeze(0),
            "negative_pixel_values": negative_inputs["pixel_values"].squeeze(0),
            "negative_input_ids": negative_inputs["input_ids"].squeeze(0),
            "negative_attention_mask": negative_inputs["attention_mask"].squeeze(0),
        }

def main():
    # --- Configuration ---
    # This would typically be loaded from config/train_config.yaml
    model_name = "google/paligemma-3b-mix-224"
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-5
    warmup_steps = 0.1 # 10% of total steps
    gradient_accumulation_steps = 1 # Simulate larger batch size if needed
    output_dir = "./output_finetune"
    margin = 0.2
    distance_metric = "cosine"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator for distributed training (handles device placement automatically)
    accelerator = Accelerator(mixed_precision="bf16") # or "fp16" for mixed precision

    # Load model and processor
    model = PaliGemmaFineTuneModel(model_name=model_name)
    processor = model.processor

    # Initialize Triplet Loss
    criterion = TripletLoss(margin=margin, distance_metric=distance_metric)

    # Create dummy dataset and DataLoader
    train_dataset = DummyWinogroundDataset(processor, num_samples=100) # Small for demo
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    # We only optimize parameters where requires_grad=True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything for acceleration (device placement, distributed training)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # --- Training Loop ---
    print("\nStarting training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Get embeddings for anchor, positive, and negative
            # Each input dict needs to be passed to model's forward separately
            anchor_embeddings = model(
                pixel_values=batch["anchor_pixel_values"],
                input_ids=batch["anchor_input_ids"],
                attention_mask=batch["anchor_attention_mask"]
            )
            positive_embeddings = model(
                pixel_values=batch["positive_pixel_values"],
                input_ids=batch["positive_input_ids"],
                attention_mask=batch["positive_attention_mask"]
            )
            negative_embeddings = model(
                pixel_values=batch["negative_pixel_values"],
                input_ids=batch["negative_input_ids"],
                attention_mask=batch["negative_attention_mask"]
            )

            # Calculate Triplet Loss
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss = loss / gradient_accumulation_steps # Scale loss for gradient accumulation

            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps # Accumulate actual loss
            progress_bar.set_postfix(loss=total_loss / (step + 1))

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.model.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma"))
    processor.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma"))
    print(f"Model saved to {os.path.join(output_dir, 'fine_tuned_paligemma')}")

if __name__ == "__main__":
    # To run this script, execute:
    # accelerate launch scripts/train.py
    # or python scripts/train.py if not using accelerate for single GPU/CPU
    main()
