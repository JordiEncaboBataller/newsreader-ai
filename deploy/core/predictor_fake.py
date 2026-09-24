"""Fake-news classifier: runs article text through a fine-tuned BERT-style
model to predict whether it's likely real or fake."""

import torch
from core.chunking_utils import predict_chunks_batched


def FAKE(text, tokenizer, model, max_len=512):
    """Predicts whether an article is real or fake and returns a dict with
    the predicted label, per-class probabilities, and the number of chunks
    the article was split into for inference."""

    # Predicts all chunks of the article in a single batched pass (see
    # chunking_utils.py) instead of a sequential chunk-by-chunk loop.
    all_logits = predict_chunks_batched(text, tokenizer, model, max_len=max_len)

    # Average the logits across all chunks before applying softmax, so
    # each part of the article contributes equally to the final prediction.
    mean_logits = torch.mean(torch.stack(all_logits), dim=0)
    probs = torch.softmax(mean_logits, dim=1)[0].numpy()

    classes = ["real", "fake"]
    pred_idx = int(probs.argmax())
    prediction = classes[pred_idx]

    result = {
        "prediction": prediction,
        "probabilities": {"real": float(probs[0]), "fake": float(probs[1])},
        "num_chunks": len(all_logits)
        }

    return result