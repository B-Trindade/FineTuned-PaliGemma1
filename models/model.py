import torch
import torch.nn as nn
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import os
import warnings

class PaliGemmaFineTune(nn.Module):
    """
    A simplified and clean wrapper for fine-tuning PaliGemma.

    This class handles:
    1. Loading the pre-trained PaliGemma model and processor.
    2. Freezing all base model parameters.
    3. Selectively unfreezing layers for fine-tuning as per the project hypothesis:
       - The multimodal projector.
       - The text embedding layer.
    4. Adding a new, trainable Layer Normalization block after the multimodal embeddings are combined.
    5. A custom forward pass to integrate the new LayerNorm.
    6. Methods for saving and loading the fine-tuned model components.
    """
    def __init__(self, model_name: str = "google/paligemma-3b-mix-224", hf_access_token: str = None):
        super().__init__()
        self.model_name = model_name
        
        # Load the base model and processor
        # We load in bfloat16 for memory efficiency.
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            token=hf_access_token
        )
        self.processor = PaliGemmaProcessor.from_pretrained(
            model_name,
            token=hf_access_token
        )

        # --- Freeze all parameters initially ---
        for param in self.model.parameters():
            param.requires_grad = False
        
        # --- Unfreeze specific layers for fine-tuning ---
        # 1. Unfreeze the Multi-Modal Projector
        if hasattr(self.model, 'multi_modal_projector'):
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
            print("Successfully unfroze 'multi_modal_projector' for fine-tuning.")
        else:
            warnings.warn("Could not find 'multi_modal_projector'. It will not be fine-tuned.")
            
        # 2. Unfreeze the Language Model's word embeddings
        if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'embed_tokens'):
            for param in self.model.language_model.model.embed_tokens.parameters():
                param.requires_grad = True
            print("Successfully unfroze 'language_model.model.embed_tokens' for fine-tuning.")
        else:
             warnings.warn("Could not find 'language_model.model.embed_tokens'. It will not be fine-tuned.")

        # --- Add the new, trainable LayerNorm block ---
        # This is a core part of the experimental hypothesis.
        embedding_dim = self.model.language_model.config.hidden_size
        self.new_layer_norm = nn.LayerNorm(embedding_dim, dtype=torch.bfloat16)
        print(f"Added a new trainable LayerNorm block with dimension {embedding_dim}.")

        # Print a summary of trainable parameters
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Prints a summary of the model's trainable parameters."""
        trainable_params = 0
        all_param = 0
        print("\n--- Trainable Parameters ---")
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"- {name}")
        print("----------------------------")
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, **kwargs):
        """
        Custom forward pass that injects the new LayerNorm block.
        This logic is necessary to modify the inputs to the language model.
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
            # Concatenate the ignore tensor with the actual text labels
            combined_labels = torch.cat((image_labels_ignore, labels), dim=1)
        else:
            combined_labels = None

        # 7. Pass the modified embeddings through the language model
        # The language model will compute the loss if labels are provided.
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        return outputs

    def save_pretrained(self, path):
        """Saves the fine-tuned components of the model."""
        os.makedirs(path, exist_ok=True)
        # Save the state dictionaries of the trainable components
        torch.save(self.model.multi_modal_projector.state_dict(), os.path.join(path, "multi_modal_projector.pth"))
        torch.save(self.model.language_model.model.embed_tokens.state_dict(), os.path.join(path, "embed_tokens.pth"))
        torch.save(self.new_layer_norm.state_dict(), os.path.join(path, "new_layer_norm.pth"))
        # Save the processor for easy loading
        self.processor.save_pretrained(path)
        print(f"Fine-tuned components saved to {path}")

    @classmethod
    def from_pretrained(cls, path, hf_access_token=None):
        """Loads a fine-tuned model from a directory."""
        # Instantiate the model with the original architecture
        # Note: The model name is hardcoded here but could be saved in a config file.
        instance = cls(model_name="google/paligemma-3b-mix-224", hf_access_token=hf_access_token)
        
        # Load the state dictionaries for the fine-tuned components
        device = next(instance.parameters()).device
        
        projector_path = os.path.join(path, "multi_modal_projector.pth")
        if os.path.exists(projector_path):
            instance.model.multi_modal_projector.load_state_dict(torch.load(projector_path, map_location=device))
            print(f"Loaded 'multi_modal_projector' state from {projector_path}")

        embed_tokens_path = os.path.join(path, "embed_tokens.pth")
        if os.path.exists(embed_tokens_path):
            instance.model.language_model.model.embed_tokens.load_state_dict(torch.load(embed_tokens_path, map_location=device))
            print(f"Loaded 'embed_tokens' state from {embed_tokens_path}")
            
        layer_norm_path = os.path.join(path, "new_layer_norm.pth")
        if os.path.exists(layer_norm_path):
            instance.new_layer_norm.load_state_dict(torch.load(layer_norm_path, map_location=device))
            print(f"Loaded 'new_layer_norm' state from {layer_norm_path}")

        print("Model loaded successfully from fine-tuned components.")
        return instance

