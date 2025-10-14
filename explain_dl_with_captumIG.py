# explain_dl_with_captumIG.py
import torch
import re
import numpy as np
from typing import List, Tuple

from captum.attr import IntegratedGradients, LayerIntegratedGradients

SEP_PATTERN = re.compile(r"[\/\.\-\_\?\=\@]")


def url_to_indices(url: str, max_len=200):
    s = str(url)[:max_len].ljust(max_len)
    return torch.tensor([min(ord(c), 127) for c in s], dtype=torch.long).unsqueeze(
        0
    )  # [1, seq]


def url_to_floats(url: str, max_len=200):
    s = str(url)[:max_len].ljust(max_len)
    arr = [ord(c) / 128.0 for c in s]
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1, seq]


def _spans_from_char_scores(
    url: str, scores: np.ndarray, k: int, max_len: int
) -> List[Tuple[str, float]]:
    s = str(url)[:max_len].ljust(max_len)
    top_idx = np.argsort(np.abs(scores))[::-1][:50]
    spans: List[Tuple[str, float]] = []
    for idx in top_idx:
        left = idx
        while left > 0 and not SEP_PATTERN.match(s[left]):
            left -= 1
        right = idx
        while right < len(s) - 1 and not SEP_PATTERN.match(s[right]):
            right += 1
        token = s[left : right + 1].strip()
        if token and token not in [t for t, _ in spans]:
            spans.append((token, float(scores[idx])))
        if len(spans) >= k:
            break
    return spans


def top_substrings_ig(model, url: str, max_len: int = 200, k: int = 3):
    """
    Returns list of (substring, importance) using:
      - LayerIntegratedGradients over model.embedding for LSTM models
      - IntegratedGradients over raw float input for CNN models
    """
    model.eval()
    device = next(model.parameters()).device

    # If model has an embedding layer (LSTM/NLP), use LayerIntegratedGradients on that layer.
    if hasattr(model, "embedding"):
        # Indices input path for LSTM-like models
        x_idx = url_to_indices(url, max_len).to(device)  # [1, seq]
        baseline_idx = torch.zeros_like(x_idx, device=device)  # zeros baseline

        lig = LayerIntegratedGradients(model, model.embedding)

        # IMPORTANT: target=None is fine for binary logits; model.forward returns [B] or [B,1]
        attributions = lig.attribute(inputs=x_idx, baselines=baseline_idx, n_steps=32)

        # attributions: [1, seq, embed_dim] -> aggregate per character by summing embedding dim
        char_scores = (
            attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        )  # [seq]
        return _spans_from_char_scores(url, char_scores, k, max_len)

    # Otherwise assume CNN-like model that consumes float sequence directly
    x = url_to_floats(url, max_len).to(device)  # [1, seq]
    x.requires_grad_(True)  # IG needs grad
    baseline = torch.zeros_like(x, device=device)

    ig = IntegratedGradients(model)

    # Model forward for CNN returns logits [B] (after .squeeze())
    attributions, _delta = ig.attribute(
        x, baseline, n_steps=32, return_convergence_delta=True
    )

    # attributions: [1, seq] -> numpy
    char_scores = attributions.squeeze(0).detach().cpu().numpy()
    return _spans_from_char_scores(url, char_scores, k, max_len)
