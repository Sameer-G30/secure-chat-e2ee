"""Shrink a BERT-family embedding table to the WordPiece ids the corpora actually use.

The word-embedding matrix is 30522 x hidden and dominates a small model's size. Most ids never
occur in chat/email text. The pruned model still accepts ORIGINAL WordPiece ids: a small int64
lookup table maps each original id to its row in the shrunken matrix (dropped ids map to [UNK]).
The browser tokenizer, fixtures, and Python evaluation therefore need no changes.
"""

# Count token usage across a corpus.
from collections import Counter

import torch
from torch import nn


class RemappedEmbedding(nn.Module):
    """Embedding that first remaps original ids onto a pruned row set."""

    def __init__(self, weight: torch.Tensor, remap: torch.Tensor, padding_idx: int | None) -> None:
        super().__init__()
        # Pruned embedding rows (kept ids only).
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=padding_idx)
        # Original id -> pruned row; a buffer so it is exported into the ONNX graph.
        self.register_buffer("remap", remap)
        # Mirror nn.Embedding attributes that HF code reads.
        self.num_embeddings = int(remap.numel())
        self.embedding_dim = int(weight.shape[1])
        self.padding_idx = padding_idx

    @property
    def weight(self) -> torch.Tensor:
        """Expose the pruned matrix like nn.Embedding.weight."""

        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for original ids via the remap table."""

        return self.embedding(self.remap[input_ids])


def count_token_usage(tokenizer, texts: list[str], max_length: int) -> Counter[int]:
    """Count how often each WordPiece id appears when tokenizing `texts`."""

    counts: Counter[int] = Counter()
    # Tokenize in chunks to bound memory.
    for start in range(0, len(texts), 4096):
        encoded = tokenizer(
            texts[start : start + 4096], truncation=True, max_length=max_length, padding=False
        )
        for ids in encoded["input_ids"]:
            counts.update(ids)
    return counts


def select_keep_ids(
    counts: Counter[int], *, min_count: int, special_ids: list[int], vocab_size: int
) -> list[int]:
    """Return sorted original ids to keep: frequent ids plus every special token."""

    keep = {token for token, count in counts.items() if count >= min_count}
    keep.update(special_ids)
    # Stay inside the original vocabulary.
    return sorted(token for token in keep if 0 <= token < vocab_size)


def prune_embeddings(model: nn.Module, keep_ids: list[int], unk_id: int) -> nn.Module:
    """Replace the word embedding with a RemappedEmbedding over `keep_ids` (in place)."""

    old = model.get_input_embeddings()
    weight = old.weight.detach()
    vocab_size = weight.shape[0]
    new_row = {original: row for row, original in enumerate(keep_ids)}
    # Dropped ids fall back to the [UNK] row.
    remap = torch.tensor(
        [new_row.get(i, new_row[unk_id]) for i in range(vocab_size)], dtype=torch.long
    )
    pruned_weight = weight[torch.tensor(keep_ids, dtype=torch.long)].clone()
    padding_idx = getattr(old, "padding_idx", None)
    new_padding = new_row.get(padding_idx) if padding_idx is not None else None
    model.set_input_embeddings(RemappedEmbedding(pruned_weight, remap, new_padding))
    return model
