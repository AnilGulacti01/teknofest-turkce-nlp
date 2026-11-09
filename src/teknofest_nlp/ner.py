"""Turkcell / Türk Telekom / Vodafone varlık tanıma (NER)."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import spacy

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "ner"

# Modelin ürettiği etiketler ve okunabilir karşılıkları
LABEL_NAMES = {
    "TURKCELL": "Turkcell",
    "TURK_TELEKOM": "Türk Telekom",
    "VODAFONE": "Vodafone",
}


class Entity(NamedTuple):
    """Metinde bulunan tek bir varlık."""

    text: str
    label: str
    start: int
    end: int


class TelecomNER:
    """Metindeki operatör adlarını bulur.

    Örnek:
        >>> ner = TelecomNER()
        >>> [e.label for e in ner.extract("Turkcell ve Vodafone kampanya başlattı.")]
        ['TURKCELL', 'VODAFONE']
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı: {model_dir}")

        # Pipeline yalnızca "ner" bileşenini içeriyor
        self.nlp = spacy.load(model_dir, disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])

    def extract(self, text: str) -> list[Entity]:
        """Metindeki varlıkları döndürür."""
        doc = self.nlp(text)
        return [
            Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            for ent in doc.ents
        ]

    @staticmethod
    def label_name(label: str) -> str:
        """Etiketin okunabilir adı (TURK_TELEKOM -> Türk Telekom)."""
        return LABEL_NAMES.get(label, label)
