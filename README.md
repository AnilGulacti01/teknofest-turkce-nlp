# TEKNOFEST Türkçe NLP Modelleri

TEKNOFEST Türkçe Doğal Dil İşleme kapsamında geliştirilmiş iki model, tek bir temiz depoda:

| Model | Ne yapar | Temel model | Boyut |
| --- | --- | --- | --- |
| **Duygu Analizi** | Metni olumlu / olumsuz olarak sınıflandırır | `dbmdz/bert-base-turkish-cased` | 442 MB |
| **Operatör NER** | Metindeki Turkcell / Türk Telekom / Vodafone geçişlerini bulur | spaCy 3.8 pipeline | 3.8 MB |

Depo hem model dosyalarını hem de onları kullanmak için gereken küçük Python paketini (`teknofest_nlp`) içerir.

## Kurulum

Duygu modelinin ağırlıkları 100 MB üzerinde olduğu için **Git LFS** ile tutuluyor:

```bash
git lfs install
git clone https://github.com/AnilGulacti01/teknofest-turkce-nlp.git
cd teknofest-turkce-nlp
```

Sonra Python ortamı (3.12 önerilir — torch ve spaCy için hazır wheel var):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`uv` ile daha hızlısı:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Kullanım

```bash
python demo.py                                   # hazır örnekler
python demo.py "Ürün harika, çok memnun kaldım." # iki model birlikte
python demo.py --sentiment "Berbat bir deneyim." # sadece duygu
python demo.py --ner "Vodafone yeni tarife açıkladı." # sadece NER
```

Kod içinden:

```python
import sys
sys.path.insert(0, "src")

from teknofest_nlp import SentimentClassifier, TelecomNER

sentiment = SentimentClassifier()
print(sentiment.predict("Kargo iki hafta gecikti, berbat bir deneyimdi.").label)
# 'negative'

ner = TelecomNER()
for entity in ner.extract("Turkcell ve Vodafone kampanya başlattı."):
    print(entity.text, "->", ner.label_name(entity.label))
# Turkcell -> Turkcell
# Vodafone -> Vodafone
```

`SentimentClassifier.predict_batch([...])` ile birden fazla metin tek seferde işlenebilir.

## Örnek çıktı

`python demo.py` komutunun gerçek çıktısı:

```
> Ürün beklediğimden çok daha iyi çıktı, kesinlikle tavsiye ederim.
  duygu  : positive (0.988)  [negative=0.012, positive=0.988]
  varlık : yok

> Kargo iki hafta gecikti ve kimse ilgilenmedi, berbat bir deneyimdi.
  duygu  : negative (1.000)  [negative=1.000, positive=0.000]
  varlık : yok
```

## Model detayları

### Duygu analizi — `models/sentiment`

- Mimari: `BertForSequenceClassification` (12 katman, 768 gizli boyut, 110M parametre)
- Tokenizer: `BertTokenizer`, cased, 32.000 kelime
- Maksimum uzunluk: 512 token
- Sınıflar: `0 = negative`, `1 = positive`

Sınıf sırası `config.json` içindeki `id2label` alanında tanımlı; bu alan sonradan eklendiği için sıra, modele verilen örnek cümlelerle doğrulandı (6/6 tutarlı: olumlu cümleler `1`, olumsuz cümleler `0`).

### Operatör NER — `models/ner`

- spaCy 3.8 pipeline (`ner` bileşeni)
- Etiketler: `TURKCELL`, `TURK_TELEKOM`, `VODAFONE`
- Vektör yok (`vectors: 0`), bu yüzden model boyutu sadece 3.8 MB

## Sınırlamalar

- **İkili sınıflandırma**: model yalnızca olumlu / olumsuz üretir. Nötr veya karışık cümleler de bir tarafa düşer (örneğin kampanya duyurusu gibi yorumsuz bir cümle olumsuz sınıflanabiliyor).
- **NER büyük/küçük harfe duyarlı**: `Turkcell` yakalanır, `turkcell` yakalanmaz. Metni modele vermeden önce büyük harfe çevirmek sonuçları iyileştirmez, çünkü model büyük harfli biçimlerle eğitilmiş.
- **Uzun metinler** 512 token'da kesilir.
- Eğitim verisi ve değerlendirme metrikleri (accuracy/F1) bu depoda yer almıyor; modeller eğitilmiş ağırlıklar olarak paylaşılıyor.

## Lisans

Kod MIT lisanslıdır — bkz. [LICENSE](LICENSE).

Model ağırlıkları, MIT lisanslı [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) üzerine fine-tune edilmiştir. NER modeli [spaCy](https://spacy.io) (MIT) ile üretilmiştir.
