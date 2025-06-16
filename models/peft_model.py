import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import warnings

class PaliGemmaFineTune(nn.Module):
    """
    A robust wrapper for fine-tuning PaliGemma, now with QLoRA support.

    This class handles:
    1. Loading the pre-trained PaliGemma model and processor.
    2. Optional 4-bit quantization of the base model (QLoRA).
    3. Optional application of LoRA adapters for parameter-efficient fine-tuning.
    4. A custom forward pass to integrate the experimental LayerNorm block.
    """
    def __init__(self, model_name: str = "google/paligemma-3b-mix-224", hf_access_token: str = None, use_qlora: bool = False, qlora_config: dict = None):
        super().__init__()
        self.model_name = model_name
        self.use_qlora = use_qlora

        # --- Configure Quantization (for QLoRA) ---
        bnb_config = None
        if self.use_qlora:
            print("QLoRA is enabled. Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # --- Load the base model ---
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # bfloat16 is used for non-quantized parts
            quantization_config=bnb_config,
            token=hf_access_token,
            device_map="auto" # Let accelerate handle device mapping
        )
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_name,
            token=hf_access_token
        )

        # --- Apply PEFT (LoRA) if enabled ---
        if self.use_qlora and qlora_config:
            print("Applying LoRA adapters to the model...")
            
            # --- OOM FIX: Replace `prepare_model_for_kbit_training` ---
            # The original utility tries to upcast layers to float32, causing OOM on ~6GB GPUs.
            # We will skip that and instead manually make the lm_head trainable in its native bfloat16.
            # This is a necessary trade-off for memory vs. potential numerical stability.
            print("Experimental OOM Fix: Preparing model for training without upcasting lm_head to float32.")
            if hasattr(self.model.language_model, 'lm_head') and isinstance(self.model.language_model.lm_head, nn.Linear):
                # By setting requires_grad, this part of the model will be trained during LoRA fine-tuning.
                # It will be trained in bfloat16, avoiding the memory spike from a float32 cast.
                self.model.language_model.lm_head.weight.requires_grad = True
                print("Set language_model.lm_head to be trainable in bfloat16.")
            # --- END OOM FIX ---
            
            lora_config = LoraConfig(
                r=qlora_config['lora_r'],
                lora_alpha=qlora_config['lora_alpha'],
                lora_dropout=qlora_config['lora_dropout'],
                target_modules=qlora_config['lora_target_modules'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA adapters applied successfully.")

            # This tells the PEFT model that the lm_head and embed_tokens weights are tied.
            # It allows the `safetensors` library to save the checkpoint correctly.
            self.model.base_model.tie_weights()
            print("Successfully tied weights for saving.")
            
        else:
            # --- FULL FINE-TUNING (The old, memory-intensive way) ---
            print("QLoRA is disabled. Using full fine-tuning for selected layers.")
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the Multi-Modal Projector
            if hasattr(self.model, 'multi_modal_projector'):
                for param in self.model.multi_modal_projector.parameters():
                    param.requires_grad = True
                print("Unfroze 'multi_modal_projector' for full fine-tuning.")
            
            # Unfreeze the Language Model's word embeddings
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'embed_tokens'):
                for param in self.model.language_model.model.embed_tokens.parameters():
                    param.requires_grad = True
                print("Unfroze 'language_model.model.embed_tokens' for full fine-tuning.")

        # --- Add the new, trainable LayerNorm block ---
        embedding_dim = self.model.language_model.config.hidden_size
        self.new_layer_norm = nn.LayerNorm(embedding_dim, dtype=torch.bfloat16)
        print(f"Added a new trainable LayerNorm block with dimension {embedding_dim}.")
        # This new layer is trainable by default

        # Print a summary of trainable parameters
        self.model.print_trainable_parameters()

    # --- FIX for AttributeError ---
    # Add methods and properties that the Trainer expects the top-level model to have.
    # These will act as pass-throughs to the underlying self.model.
    @property
    def config(self):
        return self.model.config

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Forwards the call to the underlying Hugging Face model."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    # --- END FIX ---


    def forward(self, pixel_values, input_ids, attention_mask, labels=None, **kwargs):
        """
        Custom forward pass that injects the new LayerNorm block and aligns label shapes.
        """

        # 1. Get visual embeddings from the vision tower and projector
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values.to(self.model.dtype))
        image_embeds = self.model.multi_modal_projector(vision_outputs.last_hidden_state)

        # 2. Get text embeddings from the language model's embedding layer
        text_embeds = self.model.language_model.model.embed_tokens(input_ids)

        # 3. Combine image and text embeddings
        inputs_embeds = torch.cat((image_embeds, text_embeds), dim=1)

        # 4. **Apply our custom LayerNorm block**
        inputs_embeds = self.new_layer_norm(inputs_embeds)

        # 5. Create the combined attention mask
        image_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device)
        combined_attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        # 6. Align the labels tensor with the combined embeddings
        if labels is not None:
            # Create a tensor of -100s with the same shape as the image embeddings
            image_labels_ignore = torch.full(image_embeds.shape[:-1], -100, dtype=torch.long, device=labels.device)
            combined_labels = torch.cat((image_labels_ignore, labels), dim=1)
        else:
            combined_labels = None

        # 7. Pass the modified embeddings through the language model
        # The language model will compute the loss if labels are provided.
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels
        )
        return outputs

