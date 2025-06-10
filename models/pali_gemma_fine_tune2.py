import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings

class PaliGemmaFineTuneModel(nn.Module):
    """
    A wrapper around PaliGemmaForConditionalGeneration to enable specific fine-tuning
    of chosen layers, integrate a new LayerNorm block, and apply QLoRA.
    """
    def __init__(self, model_name: str = "google/paligemma-3b-mix-224", hf_access_token: str = None,
                 use_qlora: bool = True, lora_r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.05, lora_target_modules: list = None):
        super().__init__()

        self.use_qlora = use_qlora

        # Configure 4-bit quantization if QLoRA is enabled
        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            print("Loading model with 4-bit quantization (QLoRA enabled).")
        else:
            bnb_config = None
            print("Loading model without quantization (QLoRA disabled).")

        # Load the base PaliGemma model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if not self.use_qlora else None, # Use bfloat16 if no QLoRA, else BitsAndBytes handles dtype
            token=hf_access_token,
            quantization_config=bnb_config if self.use_qlora else None,
        )
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_name,
            token=hf_access_token
        )

        # Freeze all parameters initially (PEFT will unfreeze what's needed)
        # This is mostly for clarity; PEFT's get_peft_model handles freezing.
        for param in self.model.parameters():
            param.requires_grad = False
        warnings.warn("All base model parameters are initially frozen. PEFT will manage trainable parameters.")

        # Apply PEFT (QLoRA)
        if self.use_qlora:
            # Prepare model for k-bit training (e.g., LayerNorms become float32)
            self.model = prepare_model_for_kbit_training(self.model)

            # Define LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none", # Common for QLoRA
                task_type="CAUSAL_LM", # For generative models like Gemma
                target_modules=lora_target_modules, # Specify layers to apply LoRA to
            )
            self.model = get_peft_model(self.model, lora_config)
            print("Applied QLoRA to the model.")
            self.model.print_trainable_parameters() # Show which parameters are trainable

        # --- Explicitly enable gradients for our custom architectural modifications ---
        # These layers are outside the typical LoRA targets but are part of our hypothesis.

        # 1. Linear Projection Layer (multi_modal_projector)
        if hasattr(self.model, 'multi_modal_projector') and isinstance(self.model.multi_modal_projector, nn.Module):
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
            print("Enabled fine-tuning for PaliGemma's multi_modal_projector.")
        else:
            warnings.warn("Could not find 'multi_modal_projector'. Please verify model architecture.")

        # 2. Positional Embeddings (part of the language model's embed_tokens layer)
        # This layer is often not targeted by LoRA by default, but we want to fine-tune it.
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') \
           and hasattr(self.model.language_model.model, 'embed_tokens'):
            for param in self.model.language_model.model.embed_tokens.parameters():
                param.requires_grad = True
            print("Enabled fine-tuning for PaliGemma's language_model.model.embed_tokens.")
        else:
            warnings.warn("Could not find language_model.model.embed_tokens. Positional embedding fine-tuning might not be applied.")

        # 3. New LayerNorm block (experimental)
        # This LayerNorm will be applied after the combined vision and text embeddings.
        # It needs to be trainable.
        embedding_dim = self.model.language_model.config.hidden_size
        self.new_layer_norm = nn.LayerNorm(embedding_dim)
        print(f"Added and enabled fine-tuning for a new LayerNorm with dim: {embedding_dim}.")
        # Ensure the new LayerNorm is part of the model's trainable parameters
        # (it is by default if it's a direct attribute of self, but good to be explicit)
        for param in self.new_layer_norm.parameters():
            param.requires_grad = True

        # Print all truly trainable parameters after PEFT and custom unfreezing
        print("\nFinal Trainable parameters after PEFT and custom unfreezing:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"- {name}, shape: {param.shape}")


    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        Forward pass through the PaliGemma model with specific fine-tuning modifications.
        Returns logits for VQA (CrossEntropyLoss).

        Args:
            pixel_values (torch.Tensor): Batched image pixel values.
            input_ids (torch.Tensor): Batched tokenized text input IDs (question + answer prefix).
            attention_mask (torch.Tensor): Batched attention mask for text input.
            labels (torch.Tensor, optional): Batched tokenized answer IDs for loss calculation.
                                            Set to -100 for tokens where loss should not be computed.

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithPast: Model output containing logits.
        """
        # Pass pixel_values through the vision tower to get visual features
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values)
        image_embeds = self.model.multi_modal_projector(vision_outputs.last_hidden_state)

        # Get text embeddings from the language model's embedding layer
        language_model_inputs_embeds = self.model.language_model.model.embed_tokens(input_ids)

        # Create combined attention mask for image and text
        # image_attention_mask = torch.ones(image_embeds.shape, image_embeds.shape[1],
        #                                   dtype=attention_mask.dtype, device=attention_mask.device)
        image_attention_mask = torch.ones(image_embeds.shape, image_embeds.shape[1],
                                  dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        # Concatenate image and text embeddings
        combined_inputs_embeds = torch.cat([image_embeds, language_model_inputs_embeds], dim=1)
        
        # Apply the new LayerNorm block to the combined embeddings
        normalized_combined_inputs_embeds = self.new_layer_norm(combined_inputs_embeds)

        # Pass through the language model (Gemma) with the combined and normalized embeddings
        # and the labels for causal language modeling loss.
        # The labels need to be shifted internally by the model for causal LM.
        # We pass the labels directly to the language_model.model.
        # The loss will be computed on the language model's output.
        
        # The `PaliGemmaForConditionalGeneration`'s forward method expects `inputs_embeds`
        # and `labels` directly. We need to ensure the labels are correctly aligned
        # with the combined input sequence (image tokens + text tokens).
        
        # For causal LM, labels are usually the same as input_ids, but shifted.
        # The model's internal forward handles this shifting.
        # We need to create a `labels` tensor that matches the `combined_inputs_embeds` length.
        # Loss should only be computed on the text part of the sequence.
        
        # Create labels for the combined sequence: -100 for image tokens, actual labels for text tokens
        if labels is not None:
            # -100 is the default ignore_index for CrossEntropyLoss
            image_labels = torch.full((image_embeds.shape, image_embeds.shape[1]), 
                                      -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        # Call the main model's forward method
        # Note: We are calling self.model.forward directly, not self.model.language_model.model.
        # The main model handles the full pipeline from pixel_values/input_ids to logits/loss.
        # However, since we manually constructed `combined_inputs_embeds`, we need to use
        # the `inputs_embeds` argument of the main model's forward.
        # The `PaliGemmaForConditionalGeneration`'s forward expects `pixel_values` OR `inputs_embeds`.
        # Since we are providing `inputs_embeds` (which includes image features), we should set `pixel_values=None`.
        
        # Also, the `PaliGemmaForConditionalGeneration`'s forward expects `input_ids` if `inputs_embeds` is not provided.
        # When `inputs_embeds` is provided, `input_ids` is typically not used for the LM part directly.
        # We need to ensure the `attention_mask` passed to the main model is the `combined_attention_mask`.

        # The `PaliGemmaForConditionalGeneration` forward method expects `input_ids` and `pixel_values`
        # and handles the internal concatenation. Since we are doing the concatenation ourselves
        # and applying LayerNorm, we need to pass `inputs_embeds` to the *language_model.model* directly.
        # The `PaliGemmaForConditionalGeneration` wrapper's forward method is designed for
        # direct use of `pixel_values` and `input_ids`.
        # To apply our custom LayerNorm, we must bypass the main model's default input handling.

        # Let's adjust the forward pass to directly call the language model's forward
        # after our custom LayerNorm, and then manually apply the lm_head.
        # This gives us full control over the input to the Gemma model.
        
        # Pass through the language model (Gemma) with the combined and normalized embeddings
        # and the labels for causal language modeling loss.
        # The labels need to be shifted internally by the model for causal LM.
        # We pass the labels directly to the language_model.model.
        # The loss will be computed on the language model's output.
        
        # The `PaliGemmaForConditionalGeneration`'s forward method is designed for generation.
        # It internally calls `vision_tower`, `multi_modal_projector`, and then `language_model`.
        # To insert `self.new_layer_norm`, we need to replicate parts of its forward.

        # The `language_model` attribute is `GemmaForCausalLM`. Its `forward` method takes
        # `inputs_embeds`, `attention_mask`, and `labels`.
        
        # The `GemmaForCausalLM` model will compute the loss if `labels` are provided.
        # The `labels` should correspond to the `input_ids` that are fed into the LM.
        # Since we are concatenating image_embeds and language_model_inputs_embeds,
        # the `labels` tensor must also be concatenated with -100 for image tokens.

        # Call the language model (GemmaForCausalLM) directly
        outputs = self.model.language_model(
            inputs_embeds=normalized_combined_inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels, # Pass combined labels for loss calculation
            output_hidden_states=True # Keep this for potential future use or debugging
        )
        
        # The output of GemmaForCausalLM is a CausalLMOutputWithPast, which contains `loss` and `logits`.
        return outputs

# Example Usage: (Remains the same as before, but now expects PEFT setup)
if __name__ == '__main__':
    # For testing, ensure you pass your actual token here if you are not using train.py directly
    TOKEN = ""

    # Dummy PEFT config for testing this module independently
    dummy_lora_target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = PaliGemmaFineTuneModel(
        hf_access_token=TOKEN,
        use_qlora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=dummy_lora_target_modules
    )
    processor = model.processor

    from PIL import Image
    import numpy as np

    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    dummy_text_input = ["question: What is this? answer: a cat", "question: What color? answer: blue"]

    # For labels, we need to tokenize the full sequence and then mask out non-answer parts.
    # For this dummy test, let's just use input_ids as labels for simplicity,
    # but in real VQA, labels are derived from the answer part only.
    inputs = processor(images=dummy_image, text=dummy_text_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Create dummy labels. In a real VQA setup, labels would be derived from the answer tokens.
    # For this test, let's just use input_ids as labels and mask image tokens with -100.
    # The model's internal loss calculation will handle shifting.
    dummy_labels = input_ids.clone()

    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Dummy Labels shape: {dummy_labels.shape}")

    # Pass through the model
    with torch.no_grad(): # Use no_grad for initial test to check shapes
        outputs = model(pixel_values, input_ids, attention_mask, labels=dummy_labels)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Output loss: {outputs.loss.item():.4f}")

    print("\nFinal Trainable parameters after PEFT and custom unfreezing:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}, shape: {param.shape}")