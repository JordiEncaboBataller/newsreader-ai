import torch

# Límite de bloques a analizar por artículo. Evita tiempos de espera
# desproporcionados si el scraper arrastra texto de más (menús, footers...).
# 8 bloques de ~510 tokens cubren sobradamente la longitud de un artículo
# de noticias normal (varios miles de palabras).
MAX_CHUNKS = 8


def predict_chunks_batched(text, tokenizer, model, max_len=512, max_chunks=MAX_CHUNKS):
    """Divide el texto en bloques de hasta max_len tokens (dejando hueco para
    CLS/SEP) y predice TODOS los bloques en una única pasada por el modelo
    (batch), en vez de uno a uno en un bucle secuencial.

    Antes se hacía una llamada al modelo POR CADA bloque
    (`for chunk in chunks: modelo(chunk)`), lo que multiplicaba el tiempo de
    inferencia por el número de bloques. Agrupando todos los bloques en un
    único tensor, el modelo los procesa en una sola pasada, aprovechando
    mucho mejor el paralelismo de PyTorch/BLAS y reduciendo notablemente el
    tiempo total — especialmente en artículos largos con varios bloques.

    Devuelve una lista de tensores de logits, uno por bloque (cada uno de
    forma (1, num_clases)), para mantener compatibilidad con el promediado
    posterior (torch.stack + torch.mean) que ya usan BIAS() y FAKE().
    """
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]

    step = max_len - 2  # hueco para los tokens especiales CLS y SEP

    if len(input_ids) == 0:
        token_chunks = [[]]
    else:
        token_chunks = [input_ids[i:i + step] for i in range(0, len(input_ids), step)]

    token_chunks = token_chunks[:max_chunks]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    sequences = [[cls_id] + chunk + [sep_id] for chunk in token_chunks]
    max_chunk_len = max(len(seq) for seq in sequences)

    # Padding manual para poder meter todos los bloques en un único tensor
    # batch, aunque el último bloque sea más corto que el resto.
    input_ids_batch = []
    attention_mask_batch = []
    for seq in sequences:
        pad_len = max_chunk_len - len(seq)
        input_ids_batch.append(seq + [pad_id] * pad_len)
        attention_mask_batch.append([1] * len(seq) + [0] * pad_len)

    input_ids_tensor = torch.tensor(input_ids_batch)
    attention_mask_tensor = torch.tensor(attention_mask_batch)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits  # forma: (num_bloques, num_clases)

    return [logits[i:i + 1] for i in range(logits.shape[0])]
