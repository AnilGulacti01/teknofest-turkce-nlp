#!/usr/bin/env python
"""İki modeli komut satırından denemek için basit bir demo.

Kullanım:
    python demo.py "Turkcell faturası yine zamlanmış, hiç memnun değilim."
    python demo.py --sentiment "Harika bir kampanya olmuş."
    python demo.py --ner "Vodafone ve Türk Telekom yeni tarife açıkladı."
    python demo.py            # hazır örnekleri çalıştırır
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from teknofest_nlp import SentimentClassifier, TelecomNER  # noqa: E402

DEFAULT_TEXTS = [
    "Ürün beklediğimden çok daha iyi çıktı, kesinlikle tavsiye ederim.",
    "Kargo iki hafta gecikti ve kimse ilgilenmedi, berbat bir deneyimdi.",
    "Turkcell ve Vodafone yeni kampanyalarını duyurdu, Türk Telekom da katıldı.",
]


def print_sentiment(classifier, text):
    result = classifier.predict(text)
    probs = ", ".join(f"{label}={p:.3f}" for label, p in sorted(result.probabilities.items()))
    print(f"  duygu  : {result.label} ({result.score:.3f})  [{probs}]")


def print_entities(ner, text):
    entities = ner.extract(text)
    if not entities:
        print("  varlık : yok")
        return
    formatted = ", ".join(
        f"{entity.text} -> {ner.label_name(entity.label)}" for entity in entities
    )
    print(f"  varlık : {formatted}")


def main():
    parser = argparse.ArgumentParser(description="TEKNOFEST Türkçe NLP modelleri demosu")
    parser.add_argument("text", nargs="*", help="Analiz edilecek metin")
    parser.add_argument("--sentiment", action="store_true", help="Sadece duygu analizi")
    parser.add_argument("--ner", action="store_true", help="Sadece varlık tanıma")
    args = parser.parse_args()

    texts = [" ".join(args.text)] if args.text else DEFAULT_TEXTS
    run_sentiment = args.sentiment or not args.ner
    run_ner = args.ner or not args.sentiment

    sentiment = SentimentClassifier() if run_sentiment else None
    ner = TelecomNER() if run_ner else None

    for text in texts:
        print(f"\n> {text}")
        if sentiment:
            print_sentiment(sentiment, text)
        if ner:
            print_entities(ner, text)


if __name__ == "__main__":
    main()
