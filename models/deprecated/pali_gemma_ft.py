import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import warnings

class PaliGemmaFineTuneModel(nn.Module):
    """
    A wrapper around PaliGemmaForConditionalGeneration to enable specific fine-tuning
    of chosen layers and integrate a new LayerNorm block.
    """
    def __init__(self, model_name: str = "google/paligemma-3b-mix-224"):
        super().__init__()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.processor = PaliGemmaProcessor.from_pretrained(model_name)

        # Freeze all parameters initially
        for param in self.model.parameters():
            param.requires_grad = False
        warnings.warn("All model parameters are initially frozen. Remember to unfreeze desired layers.")

        # --- Identify and enable gradients for target layers ---

        # 1. Linear Projection Layer (after SigLIP vision encoder)
        # PaliGemma uses a vision_tower (SigLIP) and then a projector to connect to Gemma.
        # This projector is usually a simple MLP or linear layer.
        # We need to find the specific projector layer.
        # Based on common VLM architectures, it's often named 'multi_modal_projector' or similar.
        # For PaliGemma, it's typically within the 'model' attribute, specifically the 'multi_modal_projector'.
        if hasattr(self.model, 'multi_modal_projector') and isinstance(self.model.multi_modal_projector, nn.Module):
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
            print("Enabled fine-tuning for PaliGemma's multi_modal_projector.")
        else:
            warnings.warn("Could not find 'multi_modal_projector'. Please verify model architecture.")

        # 2. Positional Embeddings (part of the language model's embedding layer)
        # In Gemma, RoPE (Rotary Positional Embeddings) are integrated into attention mechanism.
        # We will fine-tune the `embed_tokens` layer of the Gemma model, which is the initial
        # embedding lookup table for tokens. Changes here will indirectly affect how positional
        # information (via RoPE) interacts with token embeddings, as it's the very first
        # representation generated for the sequence.
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') \
           and hasattr(self.model.language_model.model, 'embed_tokens'):
            for param in self.model.language_model.model.embed_tokens.parameters():
                param.requires_grad = True
            print("Enabled fine-tuning for PaliGemma's language_model.model.embed_tokens.")
        else:
            warnings.warn("Could not find language_model.model.embed_tokens. Positional embedding fine-tuning might not be applied.")

        # 3. Language Branch (a portion of the Gemma model itself)
        # Fine-tuning the entire Gemma model is too much. We will fine-tune a subset,
        # for example, the last few layers of the language model or specific components.
        # For simplicity and initial experimentation, let's enable the last few decoder layers.
        # PaliGemma's language_model is a GemmaForCausalLM.
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model') \
           and hasattr(self.model.language_model.model, 'layers'):
            # Fine-tune the last 2 attention layers
            num_layers_to_finetune = 2
            gemma_layers = self.model.language_model.model.layers
            for i in range(max(0, len(gemma_layers) - num_layers_to_finetune), len(gemma_layers)):
                for param in gemma_layers[i].parameters():
                    param.requires_grad = True
                print(f"Enabled fine-tuning for Gemma Language Model layer: {i}")
            print(f"Enabled fine-tuning for the last {num_layers_to_finetune} layers of the language branch.")
        else:
            warnings.warn("Could not find language_model.model.layers. Language branch fine-tuning might not be applied.")

        # New LayerNorm block (experimental)
        # This LayerNorm will be applied after the combined vision and text embeddings,
        # before they enter the main transformer layers of the language model.
        # We need to determine the embedding dimension first.
        # PaliGemma's default hidden size for Gemma is 2048 for 3B version.
        embedding_dim = self.model.language_model.config.hidden_size
        self.new_layer_norm = nn.LayerNorm(embedding_dim)
        print(f"Added and enabled fine-tuning for a new LayerNorm with dim: {embedding_dim}.")


    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward pass through the PaliGemma model with specific fine-tuning modifications.

        Args:
            pixel_values (torch.Tensor): Batched image pixel values.
            input_ids (torch.Tensor): Batched tokenized text input IDs.
            attention_mask (torch.Tensor): Batched attention mask for text input.

        Returns:
            torch.Tensor: Pooled output embeddings for triplet loss calculation.
                          (e.g., from the last hidden state of the language model)
        """
        # Pass through the core PaliGemma model
        # We need to make sure the model returns `output_hidden_states=True`
        # to get embeddings for the triplet loss.
        # For simplicity, we'll configure it to get the last hidden state for pooling.
        # The `pixel_values` and `input_ids` are passed as in typical PaliGemma inference.

        # The PaliGemma model combines image and text internally.
        # We'll use the model's `generate` method in a different context for actual generation.
        # For triplet loss, we need embeddings from the encoder path.
        # PaliGemmaForConditionalGeneration wraps a vision encoder (SigLIP) and a language model (Gemma).
        # We need to extract the features that represent the image-text pair.

        # Directly call the model's forward pass, ensuring we get the hidden states.
        # Note: PaliGemmaForConditionalGeneration's forward method is designed for generation.
        # For embedding extraction, we need to access its internal components more directly.

        # Let's override the forward pass to get the combined input embeddings and apply LayerNorm.
        # First, get visual features
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values)
        image_embeds = self.model.multi_modal_projector(vision_outputs.last_hidden_state) # This is fine-tunable

        # Then get text embeddings
        language_model_inputs_embeds = self.model.language_model.model.embed_tokens(input_ids) # This is fine-tunable

        # Combine image and text embeddings.
        # PaliGemma typically prefixes image embeddings to text embeddings.
        # Make sure image_embeds and language_model_inputs_embeds have consistent dimensions.
        # image_embeds shape: (batch_size, num_image_tokens, hidden_size)
        # language_model_inputs_embeds shape: (batch_size, seq_len, hidden_size)

        # Create combined attention mask
        image_attention_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[1],
                                          dtype=attention_mask.dtype, device=attention_mask.device)
        combined_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        # Apply the new LayerNorm to the language model inputs BEFORE they go into the main Gemma blocks.
        # This LayerNorm is applied to the combined input sequence, including the image embeddings
        # and the textual embeddings.
        combined_inputs_embeds = torch.cat([image_embeds, language_model_inputs_embeds], dim=1)
        normalized_combined_inputs_embeds = self.new_layer_norm(combined_inputs_embeds)

        # Pass through the language model (Gemma) with the combined embeddings and mask
        # We need `output_hidden_states=True` to get the embeddings for triplet loss.
        # Ensure that `labels` are not passed, as we're not doing generation here,
        # but embedding extraction for contrastive learning.
        gemma_outputs = self.model.language_model.model(
            inputs_embeds=normalized_combined_inputs_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=True # Crucial for getting embeddings
        )

        # We'll use the last hidden state from the Gemma model as the representation
        # for the entire image-text pair. A simple mean pooling over tokens could work,
        # or taking the embedding of a specific token (e.g., CLS token if available,
        # or the last token's embedding). For simplicity, we'll mean pool.
        last_hidden_state = gemma_outputs.last_hidden_state # shape: (batch_size, seq_len, hidden_size)

        # For triplet loss, we need a single vector representation for the image-caption pair.
        # A common practice is to mean-pool the last hidden states across the sequence length.
        # Exclude padding tokens if necessary, though attention mask handles this.
        pooled_embedding = torch.mean(last_hidden_state, dim=1) # shape: (batch_size, hidden_size)

        return pooled_embedding

# Example Usage:
if __name__ == '__main__':
    # Load the model
    model = PaliGemmaFineTuneModel()
    processor = model.processor # Use the processor from the wrapped model

    # Dummy inputs for demonstration
    from PIL import Image
    import numpy as np

    # Create a dummy image (e.g., black square)
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    dummy_text_input = ["A cat playing with a ball.", "A dog chasing a car."]

    # Process inputs
    inputs = processor(images=dummy_image, text=dummy_text_input, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Pixel values shape: {pixel_values.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Pass through the model
    with torch.no_grad(): # Use no_grad for initial test to check shapes
        output_embeddings = model(pixel_values, input_ids, attention_mask)
    print(f"Output embeddings shape (pooled for triplet loss): {output_embeddings.shape}")

    # Verify which parameters are trainable
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}, shape: {param.shape}")
