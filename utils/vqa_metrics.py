import torch
from collections import defaultdict

def calculate_vqa_accuracy(predictions, references, tokenizer):
    """
    Calculates VQA accuracy based on exact match.
    For VQAv2, official evaluation is more complex (multiple answers, soft scoring).
    This provides a simple exact match for demonstration.

    Args:
        predictions (list of str): List of predicted answer strings.
        references (list of list of str): List of lists of ground truth answer strings.
                                          Each inner list contains multiple valid answers.
        tokenizer: The tokenizer used to decode predictions.

    Returns:
        dict: A dictionary containing 'accuracy' and other relevant metrics.
    """
    if len(predictions)!= len(references):
        raise ValueError("Number of predictions and references must match.")

    correct_count = 0
    total_count = len(predictions)

    for pred, refs in zip(predictions, references):
        # For VQAv2, an answer is considered correct if at least 3 out of 10 annotators
        # provided it. Here, we simplify to exact match with any reference.
        if pred.strip().lower() in [r.strip().lower() for r in refs]:
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    return {"vqa_accuracy": accuracy}

# A more robust VQA evaluation would typically use the official VQAv2 evaluation script
# which handles multiple answers and soft scoring. For a project, a simplified metric
# is often acceptable if clearly stated.

# Example of a simplified VQA scoring (as per VQAv2 paper's simplified accuracy)
def vqa_score_simplified(predicted_answer, ground_truth_answers):
    """
    Calculates a simplified VQA score for a single question.
    Score is min(1, count(predicted_answer in ground_truth_answers) / 3)
    """
    count = 0
    for gt_answer in ground_truth_answers:
        if predicted_answer.strip().lower() == gt_answer.strip().lower():
            count += 1
    return min(1, count / 3)

def calculate_vqa_scores(predictions, references):
    """
    Calculates VQA scores for a batch of predictions and references.
    """
    total_score = 0.0
    for pred, refs in zip(predictions, references):
        total_score += vqa_score_simplified(pred, refs)
    
    return {"vqa_score": total_score / len(predictions) if len(predictions) > 0 else 0.0}


if __name__ == '__main__':
    # Dummy data for testing
    preds = ["a cat", "blue", "two", "yes", "red square"]
    refs = [["a cat", "cat", "the cat"],
        ["blue", "light blue"],
        ["2", "two", "2.0"],
        ["yes", "yeah"],
        ["red circle", "a red square"]]

    # Test exact match accuracy
    acc_results = calculate_vqa_accuracy(preds, refs, None) # tokenizer not used for this simple version
    print(f"Exact Match Accuracy: {acc_results['vqa_accuracy']:.4f}")

    # Test simplified VQA score
    score_results = calculate_vqa_scores(preds, refs)
    print(f"Simplified VQA Score: {score_results['vqa_score']:.4f}")

    # Example with a perfect match
    preds_perfect = ["a cat", "blue"]
    refs_perfect = [["a cat", "cat", "the cat"],
        ["blue", "light blue"]]
    score_perfect = calculate_vqa_scores(preds_perfect, refs_perfect)
    print(f"Simplified VQA Score (Perfect): {score_perfect['vqa_score']:.4f}")

    # Example with a partial match (e.g., 1 out of 3 annotators)
    preds_partial = ["a dog"]
    refs_partial = [["a dog", "doggo", "puppy"]]
    score_partial = calculate_vqa_scores(preds_partial, refs_partial)
    print(f"Simplified VQA Score (Partial): {score_partial['vqa_score']:.4f}")