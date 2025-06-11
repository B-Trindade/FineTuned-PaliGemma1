import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import yaml
import os
import aiohttp

# Import our custom model class
from models.model import PaliGemmaFineTune

# A custom data collator to prepare batches for VQA fine-tuning.
# This is crucial for formatting the data exactly as PaliGemma expects.
class VQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        # Prepare texts, images, and answers
        images = [example['image'].convert("RGB") for example in examples]
        
        # For VQA, the prompt asks the question, and the model generates the answer.
        # The 'suffix' argument in the processor is used to create the labels for training.
        prompts = ["answer " + example['question'] for example in examples]
        answers = [example['multiple_choice_answer'] for example in examples]

        # Process the batch
        inputs = self.processor(
            text=prompts,
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=128,
            tokenize_newline_separately=False, # Important for correct label creation
        )
        return inputs

def main():
    # --- Load Configuration ---
    try:
        with open('config/train_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config/train_config.yaml not found. Please create it.")
        return

    # --- Hugging Face Access Token ---
    # Make sure to set your token in the config file.
    hf_token = config.get('hf_access_token')
    if not hf_token:
        print("WARNING: Hugging Face access token is not set in config/train_config.yaml.")
        print("This may cause issues when downloading the model or dataset.")
    
    # --- Initialize Model and Processor ---
    # Our custom model wrapper handles the architectural changes.
    print("Initializing the fine-tuning model...")
    model_wrapper = PaliGemmaFineTune(
        model_name=config['model_name'], 
        hf_access_token=hf_token
    )
    processor = model_wrapper.processor

    # --- Load and Prepare Datasets ---
    print("Loading VQAv2 dataset...")
    # Using small subsets for demonstration purposes. Adjust as needed.
    train_ds = load_dataset("HuggingFaceM4/VQAv2", split="train[:5%]", token=hf_token,
                           storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
    eval_ds = load_dataset("HuggingFaceM4/VQAv2", split="validation[:1%]", token=hf_token,
                           storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}})
    print(f"Datasets loaded. Training examples: {len(train_ds)}, Evaluation examples: {len(eval_ds)}")
    
    # The VQAv2 dataset from HuggingFaceM4 has different column names.
    # We remove the unused ones for clarity.
    train_ds = train_ds.remove_columns(['question_type', 'answers', 'question_id', 'image_id', 'answer_type'])
    eval_ds = eval_ds.remove_columns(['question_type', 'answers', 'question_id', 'image_id', 'answer_type'])

    # --- Set Up Trainer ---
    # Instantiate the data collator
    data_collator = VQADataCollator(processor)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=float(config['learning_rate']),
        warmup_ratio=0.1, # Using warmup_ratio is often more robust than fixed steps
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True, # Use bfloat16 for performance on compatible hardware
        report_to="tensorboard",
        load_best_model_at_end=True,
        remove_unused_columns=False, #* FIXED: Trainer must not delete columns before VQADataCollector processes it
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # --- Start Training ---
    print("\nStarting model fine-tuning...")
    trainer.train()
    print("Training complete.")

    # --- Save the Final Model ---
    # The trainer saves the best model automatically, but we can also save it explicitly.
    # We use our custom save method.
    final_save_path = os.path.join(config['output_dir'], "final_model")
    model_wrapper.save_pretrained(final_save_path)
    print(f"Final model components saved to {final_save_path}")

if __name__ == "__main__":
    # To run this script:
    # accelerate launch train.py
    # Or, for a single GPU:
    # python train.py
    main()
