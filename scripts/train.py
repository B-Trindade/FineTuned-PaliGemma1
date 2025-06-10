import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import random
import yaml
from PIL import Image
from io import BytesIO # To open image bytes

# Import our custom components
from models.pali_gemma_ft import PaliGemmaFineTuneModel
from utils.losses import TripletLoss
from datasets import load_dataset # For loading Winoground

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Winoground Dataset ---
class WinogroundDataset(Dataset):
    def __init__(self, processor, hf_access_token: str, split: str = "train"):
        self.processor = processor
        # Load the Winoground dataset from Hugging Face
        print(f"Loading Winoground dataset (split: {split}). This may take a while...")
        # self.dataset = load_dataset('facebook/winoground', split=split, use_auth_token=hf_access_token)
        self.dataset = load_dataset('facebook/winoground', split=split)
        print(f"Winoground dataset loaded with {len(self.dataset)} examples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Winoground items have:
        # 'image_0', 'image_1' (as dicts with 'bytes' key)
        # 'caption_0', 'caption_1' (as strings)
        # 'score_0', 'score_1', 'score_0_0', 'score_0_1', 'score_1_0', 'score_1_1'

        # Convert image bytes to PIL Image objects
        image_0_bytes = item['image_0']['bytes']
        image_1_bytes = item['image_1']['bytes']
        image_0 = Image.open(BytesIO(image_0_bytes)).convert("RGB")
        image_1 = Image.open(BytesIO(image_1_bytes)).convert("RGB")

        caption_0 = item['caption_0']
        caption_1 = item['caption_1']

        # --- Triplet Formulation Strategy for Winoground ---
        # The goal is to make the model understand which image goes with which caption,
        # especially when captions are lexically similar but semantically divergent (foils).
        # We can create two triplets per item:
        # Triplet 1: Anchor=(Image 0, Caption 0), Positive=(Image 0, Caption 0), Negative=(Image 0, Caption 1)
        # Triplet 2: Anchor=(Image 1, Caption 1), Positive=(Image 1, Caption 1), Negative=(Image 1, Caption 0)
        # This forces the model to distinguish correct image-caption pairs from incorrect (foil) ones.

        # Triplet 1
        anchor1_inputs = self.processor(images=image_0, text=caption_0, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        positive1_inputs = anchor1_inputs # The positive is conceptually the same as the anchor
        negative1_inputs = self.processor(images=image_0, text=caption_1, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Triplet 2 (Optional, but doubles the training data and reinforces distinctions)
        anchor2_inputs = self.processor(images=image_1, text=caption_1, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        positive2_inputs = anchor2_inputs
        negative2_inputs = self.processor(images=image_1, text=caption_0, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Return both triplets. DataLoader will concatenate them.
        # Ensure squeezing batch dimension as DataLoader will add its own batch dim.
        return {
            "anchor1_pixel_values": anchor1_inputs["pixel_values"].squeeze(0),
            "anchor1_input_ids": anchor1_inputs["input_ids"].squeeze(0),
            "anchor1_attention_mask": anchor1_inputs["attention_mask"].squeeze(0),
            "positive1_pixel_values": positive1_inputs["pixel_values"].squeeze(0),
            "positive1_input_ids": positive1_inputs["input_ids"].squeeze(0),
            "positive1_attention_mask": positive1_inputs["attention_mask"].squeeze(0),
            "negative1_pixel_values": negative1_inputs["pixel_values"].squeeze(0),
            "negative1_input_ids": negative1_inputs["input_ids"].squeeze(0),
            "negative1_attention_mask": negative1_inputs["attention_mask"].squeeze(0),

            "anchor2_pixel_values": anchor2_inputs["pixel_values"].squeeze(0),
            "anchor2_input_ids": anchor2_inputs["input_ids"].squeeze(0),
            "anchor2_attention_mask": anchor2_inputs["attention_mask"].squeeze(0),
            "positive2_pixel_values": positive2_inputs["pixel_values"].squeeze(0),
            "positive2_input_ids": positive2_inputs["input_ids"].squeeze(0),
            "positive2_attention_mask": positive2_inputs["attention_mask"].squeeze(0),
            "negative2_pixel_values": negative2_inputs["pixel_values"].squeeze(0),
            "negative2_input_ids": negative2_inputs["input_ids"].squeeze(0),
            "negative2_attention_mask": negative2_inputs["attention_mask"].squeeze(0),
        }

def main():
    # --- Configuration ---
    with open('config/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])

    model_name = config['model_name']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    warmup_steps = config['warmup_steps']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    output_dir = config['output_dir']
    margin = config['margin']
    distance_metric = config['distance_metric']
    hf_access_token = config['hf_access_token']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    # Removed mixed_precision for now due to user's CUDA driver warning.
    # Accelerator will automatically use CPU if CUDA is not available or compatible.
    accelerator = Accelerator()

    # Load model and processor, passing the HF access token
    model = PaliGemmaFineTuneModel(model_name=model_name)
    # Ensure the model's processor also uses the token if it's gated
    model.processor.use_auth_token = hf_access_token
    processor = model.processor

    # Initialize Triplet Loss
    criterion = TripletLoss(margin=margin, distance_metric=distance_metric)

    # Create Winoground Dataset and DataLoader
    # Using 'train' split for demonstration. In a real scenario, you'd use official splits.
    train_dataset = WinogroundDataset(processor, hf_access_token=hf_access_token, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'])

    # Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything for acceleration
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
            # Calculate loss for Triplet 1
            anchor1_embeddings = model(
                pixel_values=batch["anchor1_pixel_values"],
                input_ids=batch["anchor1_input_ids"],
                attention_mask=batch["anchor1_attention_mask"]
            )
            positive1_embeddings = model(
                pixel_values=batch["positive1_pixel_values"],
                input_ids=batch["positive1_input_ids"],
                attention_mask=batch["positive1_attention_mask"]
            )
            negative1_embeddings = model(
                pixel_values=batch["negative1_pixel_values"],
                input_ids=batch["negative1_input_ids"],
                attention_mask=batch["negative1_attention_mask"]
            )
            loss1 = criterion(anchor1_embeddings, positive1_embeddings, negative1_embeddings)

            # Calculate loss for Triplet 2
            anchor2_embeddings = model(
                pixel_values=batch["anchor2_pixel_values"],
                input_ids=batch["anchor2_input_ids"],
                attention_mask=batch["anchor2_attention_mask"]
            )
            positive2_embeddings = model(
                pixel_values=batch["positive2_pixel_values"],
                input_ids=batch["positive2_input_ids"],
                attention_mask=batch["positive2_attention_mask"]
            )
            negative2_embeddings = model(
                pixel_values=batch["negative2_pixel_values"],
                input_ids=batch["negative2_input_ids"],
                attention_mask=batch["negative2_attention_mask"]
            )
            loss2 = criterion(anchor2_embeddings, positive2_embeddings, negative2_embeddings)

            # Combine losses (e.g., average)
            loss = (loss1 + loss2) / 2.0
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
    # Make sure to create the config/train_config.yaml file first!
    main()
