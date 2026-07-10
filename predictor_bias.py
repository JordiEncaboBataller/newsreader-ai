import torch
import torch.nn.functional as F
import re
import contractions
from chunking_utils import predict_chunks_batched


def clean_text(text):

    combined = f"{text}"
    combined = contractions.fix(combined)
    combined = re.sub(r'https?://t\.co/\S+|pic\.twitter\.com/\S+', ' link_twitter ', combined)
    combined = re.sub(r'https?://\S+|www\.\S+', ' link ', combined)
    combined = re.sub(r'©.*$', ' ', combined, flags=re.MULTILINE)
    combined = re.sub(r'#\w+', ' ', combined)
    combined = re.sub(r'@\w+', ' ', combined)
    combined = re.sub(r'All rights reserved.*$', ' ', combined, flags=re.IGNORECASE|re.MULTILINE)
    # Conserva letras y tildes del español (ñ, á, é, í, ó, ú, ü) en lugar de
    # eliminarlas. Antes, la normalización NFD + el filtro solo-ASCII borraban
    # los acentos y la ñ, degradando el análisis de noticias en español.
    combined = re.sub(r'[^A-Za-zñáéíóúüÑÁÉÍÓÚÜ\s]', ' ', combined)
    combined = combined.lower()

    return re.sub(r'\s+', ' ', combined).strip()


def BIAS(texto, tokenizador, modelo, encoder, max_len=512):

    full_text = clean_text(texto)

    # Predice todos los bloques del artículo en una única pasada batched
    # (ver chunking_utils.py) en vez de un bucle secuencial bloque a bloque.
    all_logits = predict_chunks_batched(full_text, tokenizador, modelo, max_len=max_len)

    # Promediamos los logits de todos los bloques antes de aplicar softmax,
    # así cada fragmento del artículo pesa por igual en la predicción final.
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