import torch
from datasets import load_dataset
import bitsandbytes as bnb # Import bitsandbytes for the paged optimizer
from transformers import TrainingArguments, Trainer
import yaml
import os

# Import our custom model class
from models.peft_model import PaliGemmaFineTune

# A custom data collator to prepare batches for VQA fine-tuning.
class VQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        images = [example['image'].convert("RGB") for example in examples]
        prompts = ["answer " + example['question'] for example in examples]
        answers = [example['multiple_choice_answer'] for example in examples]

        inputs = self.processor(
            text=prompts,
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=128,
        )
        return inputs

def main():
    # --- Load Configuration ---
    # !! IMPORTANT: Change this to 'config_qlora.yaml' to run with the memory-efficient settings !!
    config_file = 'config/config_qlora.yaml'
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: {config_file} not found. Please create it.")
        return

    hf_token = config.get('hf_access_token')
    
    # --- Initialize Model and Processor with QLoRA settings ---
    print("Initializing the fine-tuning model...")
    model_wrapper = PaliGemmaFineTune(
        model_name=config['model_name'], 
        hf_access_token=hf_token,
        use_qlora=config.get('use_qlora', False), # Pass the QLoRA flag
        qlora_config=config.get('qlora_config', None) # Pass the QLoRA config dict
    )

    # --- Load and Prepare Datasets ---
    print("Loading VQAv2 dataset...")
    full_train_ds = load_dataset("HuggingFaceM4/VQAv2", split="train", token=hf_token)
    full_eval_ds = load_dataset("HuggingFaceM4/VQAv2", split="validation", token=hf_token)
    train_ds = full_train_ds.select(range(int(len(full_train_ds) * 0.05)))
    eval_ds = full_eval_ds.select(range(int(len(full_eval_ds) * 0.01)))
    print(f"Datasets loaded. Using {len(train_ds)} training examples and {len(eval_ds)} evaluation examples.")
    
    train_ds = train_ds.remove_columns(['question_type', 'answers', 'question_id', 'image_id', 'answer_type'])
    eval_ds = eval_ds.remove_columns(['question_type', 'answers', 'question_id', 'image_id', 'answer_type'])

    # --- Set Up Trainer ---
    data_collator = VQADataCollator(model_wrapper.processor)

    # --- OOM FIX 1: Manually create the Paged Optimizer ---
    # Find all trainable parameters in the model
    trainable_params = filter(lambda p: p.requires_grad, model_wrapper.parameters())
    # Use the paged optimizer which is designed for low-VRAM QLoRA training
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=float(config['learning_rate']))
    # ----------------------------------------------------

    # --- OOM FIX 2: Enable Gradient Checkpointing in TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        # The learning rate is now set in the optimizer, but we keep it here for reference
        learning_rate=float(config['learning_rate']),
        warmup_ratio=config['warmup_ratio'],
        logging_steps=10,
        save_strategy="epoch",
        # save_steps=30,
        save_safetensors=False,
        eval_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        gradient_checkpointing=True, # This is the key change here
    )
    # --------------------------------------------------------------------
    
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
    final_save_path = os.path.join(config['output_dir'], "final_model")
    # The trainer saves the LoRA adapters automatically.
    # We can also call save_pretrained on the model to be explicit.
    model_wrapper.model.save_pretrained(final_save_path)
    print(f"Final model adapters saved to {final_save_path}")

if __name__ == "__main__":
    main()
