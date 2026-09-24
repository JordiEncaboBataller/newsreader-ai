"""Political bias classifier: cleans article text and runs it through a
fine-tuned BERT-style model to predict left/center/right leaning."""

import torch
import torch.nn.functional as F
import re
import contractions
from core.chunking_utils import predict_chunks_batched


def clean_text(text):
    """Normalizes raw article text before classification: expands
    contractions, strips URLs, hashtags, mentions, copyright notices and
    non-alphabetic characters, and lowercases the result. Keeps Spanish
    accented letters (ñ, á, é, í, ó, ú, ü) instead of stripping them."""

    combined = f"{text}"
    combined = contractions.fix(combined)
    combined = re.sub(r'https?://t\.co/\S+|pic\.twitter\.com/\S+', ' link_twitter ', combined)
    combined = re.sub(r'https?://\S+|www\.\S+', ' link ', combined)
    combined = re.sub(r'©.*$', ' ', combined, flags=re.MULTILINE)
    combined = re.sub(r'#\w+', ' ', combined)
    combined = re.sub(r'@\w+', ' ', combined)
    combined = re.sub(r'All rights reserved.*$', ' ', combined, flags=re.IGNORECASE|re.MULTILINE)
    # Keeps Spanish letters and accents (ñ, á, é, í, ó, ú, ü) instead of
    # dropping them. Previously, NFD normalization + an ASCII-only filter
    # stripped accents and ñ, degrading analysis of Spanish-language news.
    combined = re.sub(r'[^A-Za-zñáéíóúüÑÁÉÍÓÚÜ\s]', ' ', combined)
    combined = combined.lower()

    return re.sub(r'\s+', ' ', combined).strip()


def BIAS(texto, tokenizador, modelo, encoder, max_len=512):
    """Predicts the political bias class of an article and returns a dict
    with the predicted label, per-class probabilities, and the number of
    chunks the article was split into for inference."""

    full_text = clean_text(texto)

    # Predicts all chunks of the article in a single batched pass (see
    # chunking_utils.py) instead of a sequential chunk-by-chunk loop.
    all_logits = predict_chunks_batched(full_text, tokenizador, modelo, max_len=max_len)

    # Average the logits across all chunks before applying softmax, so
    # each part of the article contributes equally to the final prediction.
    mean_logits = torch.mean(torch.stack(all_logits), dim=0)
    probs = F.softmax(mean_logits, dim=1).squeeze(0).tolist()

    classes = encoder.classes_
    pred_idx = int(torch.argmax(mean_logits, dim=1).item())
    pred_label = classes[pred_idx]

    result = {
        "prediction": pred_label,
        "probabilities": {cls: round(probs[i], 4) for i, cls in enumerate(classes)},
        "num_chunks": len(all_logits)
    }

    return result