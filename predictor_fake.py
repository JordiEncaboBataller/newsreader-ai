import torch
from chunking_utils import predict_chunks_batched


def FAKE(text, tokenizer, model, max_len=512):

    # Predice todos los bloques del artículo en una única pasada batched
    # (ver chunking_utils.py) en vez de un bucle secuencial bloque a bloque.
    all_logits = predict_chunks_batched(text, tokenizer, model, max_len=max_len)

    # Promediamos los logits de todos los bloques antes de aplicar softmax,
    # así cada fragmento del artículo pesa por igual en la predicción final.
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