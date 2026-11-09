"""BERTurk tabanlı Türkçe duygu (sentiment) sınıflandırıcı."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "sentiment"
MAX_LENGTH = 512


class SentimentResult(NamedTuple):
    """Tek bir metnin duygu tahmini."""

    label: str
    score: float
    probabilities: dict[str, float]


class SentimentClassifier:
    """Metni olumlu / olumsuz olarak sınıflandırır.

    Örnek:
        >>> clf = SentimentClassifier()
        >>> clf.predict("Ürün beklediğimden çok daha iyi çıktı.").label
        'positive'
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | None = None):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # config.json içindeki id2label her zaman int anahtarla gelmeyebilir
        self.id2label = {int(i): label for i, label in self.model.config.id2label.items()}

    @torch.inference_mode()
    def predict(self, text: str) -> SentimentResult:
        """Tek bir metin için tahmin döndürür."""
        batch = self._prepare([text])
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probs))

        return SentimentResult(
            label=self.id2label[index],
            score=float(probs[index]),
            probabilities={self.id2label[i]: float(p) for i, p in enumerate(probs)},
        )

    @torch.inference_mode()
    def predict_batch(self, texts: Sequence[str]) -> list[SentimentResult]:
        """Birden fazla metni tek seferde sınıflandırır."""
        texts = list(texts)
        if not texts:
            return []

        batch = self._prepare(texts, padding=True)
        logits = self.model(**batch).logits
        probs = torch.softmax(logits, dim=-1)

        results = []
        for row in probs:
            index = int(torch.argmax(row))
            results.append(
                SentimentResult(
                    label=self.id2label[index],
                    score=float(row[index]),
                    probabilities={self.id2label[i]: float(p) for i, p in enumerate(row)},
                )
            )
        return results

    def _prepare(self, texts: list[str], padding: bool = False) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=padding,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}
