"""Shared text-chunking and batched-inference utilities for the BIAS and
FAKE classifiers, both of which use the same BERT-style tokenizer/model
pattern and need to split long articles into model-sized chunks."""

import torch

# Max number of chunks analyzed per article. Prevents disproportionate
# wait times if the scraper pulls in extra text (menus, footers...).
# 8 chunks of ~510 tokens comfortably cover the length of a typical news
# article (several thousand words).
MAX_CHUNKS = 8


def predict_chunks_batched(text, tokenizer, model, max_len=512, max_chunks=MAX_CHUNKS):
    """Splits the text into chunks of up to max_len tokens (leaving room for
    the CLS/SEP special tokens) and predicts ALL chunks in a single forward
    pass through the model (batched), instead of one at a time in a
    sequential loop.

    Previously, the model was called ONCE PER CHUNK
    (`for chunk in chunks: model(chunk)`), which multiplied inference time
    by the number of chunks. By stacking all chunks into a single tensor,
    the model processes them in one pass, taking much better advantage of
    PyTorch/BLAS parallelism and noticeably reducing total time —
    especially for long articles with several chunks.

    Returns a list of logit tensors, one per chunk (each of shape
    (1, num_classes)), to stay compatible with the averaging step
    (torch.stack + torch.mean) already used by BIAS() and FAKE().
    """
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"]

    step = max_len - 2  # room for the special CLS and SEP tokens

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

    # Manual padding so every chunk can be stacked into a single batch
    # tensor, even if the last chunk is shorter than the rest.
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
        logits = outputs.logits  # shape: (num_chunks, num_classes)

    return [logits[i:i + 1] for i in range(logits.shape[0])]