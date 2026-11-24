"""Test that align() keeps each input segment as one sentence."""

from types import SimpleNamespace

import numpy as np
import torch

from whisperx.alignment import align


def test_segment_with_sentence_punctuation_stays_one_segment():
    dictionary = {"<pad>": 0, "あ": 1, "い": 2}
    model = lambda values: SimpleNamespace(logits=torch.zeros(1, 49, len(dictionary)))
    meta = {"dictionary": dictionary, "language": "ja", "type": "huggingface"}
    segment = {"start": 0.0, "end": 1.0, "text": " あ! い. あ "}

    out = align([segment], model, meta, np.zeros(16000, dtype=np.float32), "cpu")

    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"] == "あ! い. あ"
    assert [w["word"] for w in out["segments"][0]["words"]] == ["あ", "!", "い", ".", "あ"]
