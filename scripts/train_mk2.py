import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
import os
import random
import yaml
from PIL import Image
from io import BytesIO
import json # For VQAv2 answers

# Import our custom components
from models.pali_gemma_finetune_model import PaliGemmaFineTuneModel
from utils.vqa_metrics import calculate_vqa_scores # Using simplified VQA score
from utils.winoground_metrics import calculate_winoground_scores
from datasets import load_dataset # For loading VQAv2 and Winoground

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    os.environ = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- VQA Dataset for Training and Validation ---
class VQADataset(Dataset):
    def __init__(self, processor, hf_access_token: str, split: str, max_seq_length: int):
        self.processor = processor
        self.max_seq_length = max_seq_length
        
        # Load VQAv2 dataset. Use a subset for training if specified in split.
        print(f"Loading VQAv2 dataset (split: {split}). This may take a while...")
        self.dataset = load_dataset('HuggingFaceM4/VQAv2', split=split, use_auth_token=hf_access_token)
        print(f"VQAv2 dataset loaded with {len(self.dataset)} examples.")

        # Ensure tokenizer has a pad_token
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.add_special_tokens({'pad_token': ''})
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.convert_tokens_to_ids('')
            self.processor.model.resize_token_embeddings(len(self.processor.tokenizer))
            print(f"Added token to tokenizer and resized embeddings. New vocab size: {len(self.processor.tokenizer)}")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image'].convert("RGB")
        question = item['question']
        # VQAv2 has 10 ground truth answers. We pick the first one for simplicity as the target.
        # For more robust VQA, you might sample one or use all for evaluation.
        answers = item['answers']
        ground_truth_answer = answers # Use the first answer as the target for training

        # PaliGemma expects input in a specific prompt format for VQA
        # Example: "question: What is this? answer:"
        # The model then generates the answer.
        prompt_text = f"question: {question} answer:"
        full_text = f"question: {question} answer: {ground_truth_answer}"

        # Tokenize the full text (prompt + answer)
        # The labels will be derived from this, masking out the prompt part.
        
        # Process image and text
        # The processor handles image resizing and normalization.
        # It also tokenizes the text.
        
        # For VQA, we want the model to generate the answer.
        # The labels for causal LM should be the answer tokens, with prompt tokens masked.
        
        # Tokenize the prompt and the full text separately to identify answer tokens
        prompt_inputs = self.processor.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        
        full_inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )

        # Create labels: -100 for prompt tokens and padding, actual token IDs for answer tokens
        labels = full_inputs["input_ids"].clone()
        # Mask prompt tokens and padding tokens in labels with -100
        labels[0, :prompt_inputs["input_ids"].shape[1]] = -100
        
        # If the full_text was truncated, ensure labels beyond truncation point are -100
        # This is handled by padding="max_length" and truncation=True, but good to be aware.
        
        # Prepare inputs for the model
        model_inputs = self.processor(
            images=image,
            text=prompt_text, # Only prompt text is passed to the processor for input_ids
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )

        return {
            "pixel_values": model_inputs["pixel_values"].squeeze(0),
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "question": question, # Keep for evaluation
            "answers": answers # Keep for evaluation
        }

# --- Winoground Dataset for Evaluation Only ---
class WinogroundEvaluationDataset(Dataset):
    def __init__(self, processor, hf_access_token: str):
        self.processor = processor
        print(f"Loading Winoground dataset (split: 'test') for evaluation.")
        # Winoground only has a 'test' split
        self.dataset = load_dataset('facebook/winoground', split="test")
        print(f"Winoground dataset loaded with {len(self.dataset)} examples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert image bytes to PIL Image objects
        image_0 = Image.open(BytesIO(item['image_0']['bytes'])).convert("RGB")
        image_1 = Image.open(BytesIO(item['image_1']['bytes'])).convert("RGB")
        caption_0 = item['caption_0']
        caption_1 = item['caption_1']

        # For evaluation, we return the raw components to be processed by calculate_winoground_scores
        return {
            "image_0": image_0,
            "image_1": image_1,
            "caption_0": caption_0,
            "caption_1": caption_1,
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
    hf_access_token = config['hf_access_token']
    use_qlora = config['use_qlora']
    lora_r = config['lora_r']
    lora_alpha = config['lora_alpha']
    lora_dropout = config['lora_dropout']
    lora_target_modules = config['lora_target_modules']
    max_seq_length = config['max_seq_length']
    eval_steps = config['eval_steps']

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load model and processor
    model_wrapper = PaliGemmaFineTuneModel(
        model_name=model_name,
        hf_access_token=hf_access_token,
        use_qlora=use_qlora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules
    )
    processor = model_wrapper.processor # Get the processor from the wrapper

    # Create VQA Datasets and DataLoaders
    # Using train[:10%] for training as per Colab notebook suggestion
    train_split_str = "train[:10%]" # Or "train" for full dataset
    val_split_str = "validation"

    train_dataset = VQADataset(processor, hf_access_token=hf_access_token, split=train_split_str, max_seq_length=max_seq_length)
    val_dataset = VQADataset(processor, hf_access_token=hf_access_token, split=val_split_str, max_seq_length=max_seq_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])

    # Create Winoground Evaluation Dataset and DataLoader
    winoground_eval_dataset = WinogroundEvaluationDataset(processor, hf_access_token=hf_access_token)
    # Winoground evaluation is typically done item by item or in small batches for specific scoring,
    # so a DataLoader might not be strictly necessary for `calculate_winoground_scores` if it handles iteration.
    # For now, we'll pass the dataset directly.

    # Optimizer
    # Only optimize parameters where requires_grad=True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=learning_rate)

    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_steps)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything for acceleration
    model_wrapper, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model_wrapper, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # --- Initial Zero-Shot Evaluation (Baselines) ---
    print("\n--- Performing Initial Zero-Shot Baselines ---")
    model_wrapper.eval()
    # VQAv2 Zero-Shot
    vqa_preds_baseline =
    vqa_refs_baseline =
    print("Evaluating VQAv2 Zero-Shot Baseline...")
    for batch in tqdm(val_dataloader, desc="VQAv2 Zero-Shot"):
        with torch.no_grad():
            # Generate answers for VQA. We need to pass only the prompt.
            # The model's generate method is used for inference.
            # Ensure the model is unwrapped for generate if it's a PEFT model.
            generated_ids = accelerator.unwrap_model(model_wrapper).model.generate(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"], # This is the prompt part
                attention_mask=batch["attention_mask"],
                max_new_tokens=20, # Max tokens for answer generation
                do_sample=False, # Greedy decoding
                pad_token_id=processor.tokenizer.pad_token_id # Ensure pad token is handled
            )
            # Decode generated IDs. Skip the input prompt tokens.
            # The generated_ids include the input_ids (prompt).
            # We need to slice to get only the generated answer.
            # The length of the prompt is `batch["input_ids"].shape[1]`.
            generated_answers = processor.batch_decode(generated_ids[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            
            vqa_preds_baseline.extend(generated_answers)
            vqa_refs_baseline.extend(batch["answers"]) # batch["answers"] is a list of lists of strings

    vqa_baseline_scores = calculate_vqa_scores(vqa_preds_baseline, vqa_refs_baseline)
    print(f"VQAv2 Zero-Shot Baseline Score: {vqa_baseline_scores['vqa_score']:.4f}")

    # Winoground Zero-Shot
    print("Evaluating Winoground Zero-Shot Baseline...")
    winoground_baseline_scores = calculate_winoground_scores(
        model=model_wrapper,
        processor=processor,
        winoground_dataset=winoground_eval_dataset,
        device=device # Pass accelerator's device
    )
    print(f"Winoground Zero-Shot Baseline Scores: {winoground_baseline_scores}")
    
    # --- Training Loop ---
    print("\nStarting training...")
    model_wrapper.train()
    total_steps = len(train_dataloader) * num_epochs
    eval_interval = int(total_steps * eval_steps) if eval_steps < 1 else eval_steps
    if eval_interval == 0: eval_interval = 1 # Ensure at least one eval step

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            outputs = model_wrapper(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"] # Labels for CrossEntropyLoss
            )
            
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            global_step += 1
            progress_bar.set_postfix(loss=total_loss / (step + 1))

            # --- Periodic VQA Validation Evaluation ---
            if global_step % eval_interval == 0:
                accelerator.print(f"\n--- Evaluating at step {global_step} ---")
                model_wrapper.eval()
                vqa_preds_val =
                vqa_refs_val =
                for val_batch in tqdm(val_dataloader, desc="VQAv2 Validation"):
                    with torch.no_grad():
                        generated_ids = accelerator.unwrap_model(model_wrapper).model.generate(
                            pixel_values=val_batch["pixel_values"],
                            input_ids=val_batch["input_ids"],
                            attention_mask=val_batch["attention_mask"],
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id
                        )
                        generated_answers = processor.batch_decode(generated_ids[:, val_batch["input_ids"].shape[1]:], skip_special_tokens=True)
                        vqa_preds_val.extend(generated_answers)
                        vqa_refs_val.extend(val_batch["answers"])
                
                vqa_val_scores = calculate_vqa_scores(vqa_preds_val, vqa_refs_val)
                accelerator.print(f"VQAv2 Validation Score: {vqa_val_scores['vqa_score']:.4f}")
                model_wrapper.train() # Switch back to train mode

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

    # --- Final Evaluation on Winoground ---
    print("\n--- Performing Final Winoground Evaluation ---")
    model_wrapper.eval()
    winoground_final_scores = calculate_winoground_scores(
        model=model_wrapper,
        processor=processor,
        winoground_dataset=winoground_eval_dataset,
        device=device
    )
    print(f"Winoground Final Scores: {winoground_final_scores}")

    # Save the fine-tuned model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model_wrapper)
    # If using PEFT, save the adapter model, not the full model
    if use_qlora:
        unwrapped_model.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma_peft"))
        processor.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma_peft"))
        print(f"PEFT adapters and processor saved to {os.path.join(output_dir, 'fine_tuned_paligemma_peft')}")
    else:
        unwrapped_model.model.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma"))
        processor.save_pretrained(os.path.join(output_dir, "fine_tuned_paligemma"))
        print(f"Full model and processor saved to {os.path.join(output_dir, 'fine_tuned_paligemma')}")

if __name__ == "__main__":
    # To run this script, execute:
    # accelerate launch scripts/train.py
    # Make sure to create the config/train_config.yaml file first!
    main()