import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss for learning embeddings.
    It enforces that the distance between an anchor and a positive example
    is smaller than the distance between the anchor and a negative example
    by at least a given margin.
    """
    def __init__(self, margin: float = 0.2, distance_metric: str = "cosine"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
        self.distance_metric = distance_metric

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Calculates the triplet loss.

        Args:
            anchor (torch.Tensor): Embedding of the anchor example.
            positive (torch.Tensor): Embedding of the positive example.
            negative (torch.Tensor): Embedding of the negative example.

        Returns:
            torch.Tensor: The computed triplet loss.
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance squared for computational efficiency
            d_pos = F.pairwise_distance(anchor, positive, p=2)
            d_neg = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            d_pos = 1 - F.cosine_similarity(anchor, positive)
            d_neg = 1 - F.cosine_similarity(anchor, negative)

        loss = torch.relu(d_pos - d_neg + self.margin)
        return loss.mean()

# Example Usage:
if __name__ == '__main__':
    # Dummy embeddings
    anchor_emb = torch.randn(10, 128)
    positive_emb = torch.randn(10, 128)
    negative_emb = torch.randn(10, 128)

    # Make positive closer to anchor
    positive_emb = anchor_emb + 0.1 * torch.randn(10, 128)
    # Make negative farther from anchor
    negative_emb = anchor_emb + 2.0 * torch.randn(10, 128)


    triplet_loss_euclidean = TripletLoss(margin=0.5, distance_metric="euclidean")
    loss_e = triplet_loss_euclidean(anchor_emb, positive_emb, negative_emb)
    print(f"Euclidean Triplet Loss: {loss_e.item():.4f}")

    triplet_loss_cosine = TripletLoss(margin=0.2, distance_metric="cosine")
    loss_c = triplet_loss_cosine(anchor_emb, positive_emb, negative_emb)
    print(f"Cosine Triplet Loss: {loss_c.item():.4f}")

    # Test case where negative is already far enough
    anchor_emb_far = torch.randn(1, 128)
    positive_emb_far = anchor_emb_far + 0.05 * torch.randn(1, 128)
    negative_emb_far = anchor_emb_far + 5.0 * torch.randn(1, 128) # Very far

    loss_zero_margin = triplet_loss_cosine(anchor_emb_far, positive_emb_far, negative_emb_far)
    print(f"Cosine Triplet Loss (Negative already far enough): {loss_zero_margin.item():.4f}")
