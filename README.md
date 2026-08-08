# SMS Spam Detection — TF-IDF Baseline vs. Fine-Tuned Transformer

Classifies SMS text messages as **spam** or **ham** (legitimate), comparing a classic NLP baseline against a fine-tuned transformer.

**Live demo:** [spamdetectionnlp.streamlit.app](https://spamdetectionnlp-gqbji7rzap7qnjaxahjmzy.streamlit.app/)

---

## Dataset

[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,574 real SMS messages labeled ham/spam (5,169 after removing duplicates). Class distribution: ~87% ham / ~13% spam.

## Project structure

```
spam-detection-nlp/
├── app.py                                    # Streamlit deployment (baseline model)
├── data/
│   └── SMSSpamCollection
├── models/
│   ├── logreg_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   ├── 01_baseline_tfidf_logreg.ipynb
│   └── 02_transformer_comparison.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Approach

**1. EDA** — confirmed class imbalance (~87/13), found spam messages average ~139 characters vs. ~71 for ham (nearly double, driven by spam's need to fit a hook, offer, and call-to-action in one message), and removed 403 duplicate rows before splitting to avoid data leakage between train and test sets.

**2. Feature engineering (baseline)** — extracted `num_exclamations`, `has_currency_symbol`, `pct_uppercase`, and `num_digits` from raw text before cleaning, so punctuation/capitalization signal wasn't lost during normalization. Text itself was lowercased, stripped of punctuation and digits, stopword-filtered, and stemmed, then vectorized with TF-IDF (3,000 features) and combined with the four numeric features.

**3. Baseline model** — Logistic Regression with `class_weight='balanced'` to counter class imbalance, evaluated on precision/recall/F1 rather than accuracy (a trivial "always predict ham" rule would already score ~87% accuracy while catching zero spam).

**4. Transformer model** — fine-tuned `distilbert-base-uncased` on raw, unprocessed text (no stemming/stopword removal — transformers use grammar and word context directly, which classic preprocessing would destroy). Trained 3 epochs; best checkpoint (epoch 1) selected by F1 score, since epochs 2–3 showed mild overfitting (validation F1 declined slightly).

**5. Deployment** — the baseline model is deployed via Streamlit, chosen over the transformer for its smaller footprint and near-instant inference; the transformer's ~268MB model size and slower CPU inference made it a poor fit for a lightweight public demo.

## Results

| Metric | Baseline (TF-IDF + LogReg) | Transformer (DistilBERT) |
|---|---|---|
| Precision (spam) | 0.96 | 0.976 |
| Recall (spam) | 0.92 | 0.947 |
| F1 (spam) | 0.94 | 0.961 |
| Training time | <1 sec | ~35 min (CPU) |

The transformer outperforms the baseline on every metric, at a substantial training-cost tradeoff.

### Error analysis

Reading the actual misclassified messages (not just the metrics) showed *why* the transformer wins: the baseline's engineered features (exclamation counts, currency symbols, uppercase ratio) only catch "loud" spam ("FREE!!! CALL NOW!!!"). Quieter, conversational-style spam — e.g. *"Babe: U want me dont u baby!..."*, *"Dear Voucher Holder, To claim..."* — has none of those markers and relies entirely on vocabulary and context, which the transformer captures and the baseline's numeric features can't.

Both models share a genuine blind spot on ringtone-subscription spam (e.g. *"RECPT 1/3. You have ordered a Ringtone..."*), which reads closer to a legitimate order confirmation than obvious spam — likely a case of real label ambiguity in the dataset rather than a pure model weakness.

## Limitations

- **Domain generalization**: both models were trained on a single, older (2011), UK-centric SMS dataset. Manual testing showed the baseline fails on generic modern ad-style phrasing (e.g. "SALE!! BUY IT NOW!!!") that doesn't resemble the dataset's prize/lottery/ringtone-scam style of spam — a known limitation of models trained on one narrow data distribution.
- **`distilbert-base-uncased` can't use capitalization as a signal** (it was pretrained on lowercased text), unlike the baseline's explicit `pct_uppercase` feature — a deliberate tradeoff for using the smaller, faster "uncased" variant.
- The deployed app uses the baseline model only; the transformer's stronger results aren't reflected in the live demo.

## Running locally

```bash
git clone https://github.com/g30613740/spam_detection_nlp.git
cd spam_detection_nlp
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, NLTK, HuggingFace Transformers, PyTorch, Streamlit


## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Author: @g30613740 Philip K.

## GitHub: https://github.com/g30613740/spam_detection_nlp