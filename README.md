# Sentiment Analysis of YouTube Gaming Comments (Maze Dataset)

## 📌 Project Overview
This project presents a comprehensive Natural Language Processing (NLP) pipeline designed to classify the sentiment (Positive, Negative, Neutral) of highly informal, noisy, and slang-heavy YouTube gaming comments for the game "Maze". 

The research compares traditional machine learning baseline models against state-of-the-art contextual Transformers. It explores the impact of specialized preprocessing, the mathematical transition from sparse to dense embeddings, and the performance differences between frozen feature extraction and end-to-end deep fine-tuning.

---

## 🏗️ Architecture & Pipeline

### 1. Advanced NLP Preprocessing (C & Python)
A domain-aware preprocessing pipeline was built to handle internet slang, emojis, and noise. To test the efficacy of these steps, an **Ablation Study** was conducted across 5 variations:
* **Var 1 (`all_preprocessing`)**: Normalization, Emoji Mapping (e.g., `emoji_positive`), Hashtag retention, Casing preservation, Elongation compression (`__ELONG__`), and custom punctuation tokenization.
* **Var 2 (`normalising_casing_space`)**: Minimal cleaning.
* **Var 3 (`no_emoji_handling`)**: Emojis ignored.
* **Var 4 (`no_hash_no_elong`)**: Hashtags and elongations ignored.
* **Var 5 (`all_lowercased`)**: Full preprocessing, but lowercased (Winner for Base Models).

### 2. Phase 1: Baseline Models & Embeddings
To establish a "Spectrum of Complexity", the dataset was evaluated using a progression of linguistic representations:
* **Sparse Embeddings:** TF-IDF (capped at 5000 features to prevent the Curse of Dimensionality and overfitting).
* **Static Dense Embeddings (CBOW Mean Pooling):** Word2Vec (local context), GloVe (global context), FastText (sub-word tokenization for typos).
* **Contextual Dense Embeddings (Frozen):** SBERT (Sentence-BERT Siamese Network), RoBERTa-base (`[CLS]` token extraction).

**Base Classifiers Evaluated:**
* **Naive Bayes:** Probabilistic baseline.
* **LinearSVC:** Geometric hyperplane optimizer (Highly effective with TF-IDF).
* **XGBoost:** Non-linear tree ensemble (Histogram-optimized for CPU speed).
* **MLP:** Standard Feed-Forward Neural Network.
* **Custom Ensembles:** Meta-Learner Stacking (Logistic Regression meta-model) and Weighted Soft Voting (SVC 3x, MLP 1x, XGB 1x).

### 3. Phase 2: Deep Learning & Fine-Tuning
The frozen base models hit a performance ceiling due to the complexity of sarcasm and contextual dependencies (polysemy). 
* **Model:** End-to-end fine-tuning of `roberta-base`.
* **Synthetic Augmentation:** Prompt-driven LLM augmentation was used to generate varied training data using directives like `register_slang`, `register_analytical`, `temporal_framing`, and `aspect_rotation`.
* **Results:** Accuracy leaped from the ~82% baseline ceiling to a massive **97.62%**.

### 4. Post-Processing & Error Analysis
To mitigate the inherent ambiguity of the "Neutral" class and domain-specific edge cases, two advanced post-processing protocols were implemented:
1.  **Confidence Thresholding (Probability Calibration):** Applied to the raw softmax logits of the fine-tuned RoBERTa model. If the model predicted Positive/Negative with less than 65% confidence, it was conservatively overridden to Neutral, mimicking human hesitation.
2.  **Rule-Based Heuristics:** Scripts designed to catch specific logic traps (e.g., The "Event vs. Opinion" trap, the "Cynical But" trap).

---

## 📊 Key Findings & Results

* **Metric:** Models were evaluated primarily on **Weighted F1-Score** to mathematically account for the natural class imbalance of the real-world dataset.
* **Base Model Winner:** `var_5_all_lowercased` + **LinearSVC** using RoBERTa embeddings → **82.54% Accuracy / 82.47 F1**.
* **Fine-Tuning Winner:** `var_1_all_preprocessing` with **Fine-Tuned RoBERTa** → **97.62% Accuracy / 97.60 F1**.
* **SBERT vs. RoBERTa (Frozen):** SBERT vastly outperformed frozen RoBERTa with Naive Bayes (77.38% vs 36.11%), proving that Siamese-pooled sentence embeddings are far more classifier-friendly than raw `[CLS]` extraction without fine-tuning.

---

## 🔬 Detailed Metrics — RoBERTa & BERT

> All test metrics below are **Weighted** (Precision, Recall, F1) evaluated on the held-out test split (252 samples).

### Phase 1: Frozen RoBERTa Embeddings — Test Set Performance

| Variation | Classifier | Accuracy | Precision | Recall | F1 Score |
|:---|:---|:---:|:---:|:---:|:---:|
| `var_1_all_preprocessing` | Naive Bayes | 36.11 | 42.37 | 36.11 | 35.91 |
| | LinearSVC | **80.56** | **80.91** | **80.56** | **80.41** |
| | MLP NeuralNet | 76.98 | 77.16 | 76.98 | 77.01 |
| | XGBoost | 73.81 | 73.79 | 73.81 | 73.79 |
| `var_2_normalising_casing_space` | Naive Bayes | 36.90 | 44.17 | 36.90 | 37.08 |
| | LinearSVC | **79.37** | **79.91** | **79.37** | **79.26** |
| | MLP NeuralNet | 67.06 | 67.84 | 67.06 | 67.11 |
| | XGBoost | 69.84 | 70.39 | 69.84 | 69.83 |
| `var_3_all_no_emoji` | Naive Bayes | 36.51 | 44.88 | 36.51 | 36.88 |
| | LinearSVC | **78.97** | **79.77** | **78.97** | **78.86** |
| | MLP NeuralNet | 68.25 | 68.32 | 68.25 | 68.28 |
| | XGBoost | 70.63 | 70.80 | 70.63 | 70.54 |
| `var_4_all_no_hash_no_elong` | Naive Bayes | 36.11 | 42.37 | 36.11 | 35.91 |
| | LinearSVC | **80.16** | **80.56** | **80.16** | **80.01** |
| | MLP NeuralNet | 70.24 | 70.77 | 70.24 | 69.91 |
| | XGBoost | 73.81 | 74.16 | 73.81 | 73.76 |
| `var_5_all_lowercased` ⭐ | Naive Bayes | 41.27 | 46.78 | 41.27 | 40.85 |
| | LinearSVC | **82.54** | **83.46** | **82.54** | **82.47** |
| | MLP NeuralNet | 77.38 | 77.22 | 77.38 | 77.13 |
| | XGBoost | 73.81 | 73.95 | 73.81 | 73.77 |

> ⭐ **Best frozen-RoBERTa result:** `var_5_all_lowercased` + LinearSVC at **82.54% Accuracy**. Lowercasing acted as dimensionality reduction — it collapsed casing variations (e.g., "AMAZING", "Amazing", "amazing") into a single representation, making the linear decision boundary cleaner for a frozen embedding space.

---

### Phase 1: Frozen BERT Embeddings — Test Set Performance

| Variation | Classifier | Accuracy | Precision | Recall | F1 Score |
|:---|:---|:---:|:---:|:---:|:---:|
| `var_1_all_preprocessing` | Naive Bayes | 56.35 | 58.84 | 56.35 | 55.37 |
| | LinearSVC | **66.27** | **66.54** | **66.27** | **66.36** |
| | MLP NeuralNet | 61.51 | 61.83 | 61.51 | 61.50 |
| | XGBoost | 60.71 | 61.15 | 60.71 | 60.77 |
| `var_2_normalising_casing_space` | Naive Bayes | 55.16 | 57.49 | 55.16 | 55.15 |
| | LinearSVC | **63.10** | **63.60** | **63.10** | **63.18** |
| | MLP NeuralNet | 53.17 | 54.64 | 53.17 | 53.10 |
| | XGBoost | 63.10 | 62.99 | 63.10 | 62.92 |
| `var_3_all_no_emoji` | Naive Bayes | 56.35 | 58.55 | 56.35 | 56.32 |
| | LinearSVC | **64.29** | **64.82** | **64.29** | **64.33** |
| | MLP NeuralNet | 56.35 | 56.60 | 56.35 | 56.37 |
| | XGBoost | 63.89 | 64.29 | 63.89 | 63.89 |
| `var_4_all_no_hash_no_elong` | Naive Bayes | 58.33 | 60.52 | 58.33 | 57.50 |
| | LinearSVC | **65.48** | **65.74** | **65.48** | **65.54** |
| | MLP NeuralNet | 55.95 | 56.83 | 55.95 | 56.03 |
| | XGBoost | 61.51 | 62.55 | 61.51 | 61.60 |
| `var_5_all_lowercased` | Naive Bayes | 57.54 | 58.96 | 57.54 | 57.22 |
| | LinearSVC | **69.44** | **69.72** | **69.44** | **69.52** |
| | MLP NeuralNet | 56.75 | 57.10 | 56.75 | 56.46 |
| | XGBoost | 65.48 | 65.91 | 65.48 | 65.62 |

> **Best frozen-BERT result:** `var_5_all_lowercased` + LinearSVC at **69.44% Accuracy** — a full **13.10 percentage points below** the equivalent RoBERTa configuration. BERT's smaller pre-training corpus and lack of dynamic masking contributed to weaker contextual representations for this domain.

---

### Validation Accuracy Comparison — BERT vs. RoBERTa (Training Phase)

| Variation | Embedding | Naive Bayes | LinearSVC | MLP NeuralNet | XGBoost |
|:---|:---|:---:|:---:|:---:|:---:|
| `var_1_all_preprocessing` | **BERT** | 57.25% | 65.82% | 60.73% | 63.71% |
| | **RoBERTa** | 50.75% | **69.61%** | 66.90% | 67.72% |
| `var_2_normalising_casing_space` | **BERT** | 57.94% | 65.71% | 60.23% | 64.21% |
| | **RoBERTa** | 51.52% | **70.22%** | 66.63% | 67.07% |
| `var_3_all_no_emoji` | **BERT** | 57.15% | 65.74% | 59.46% | 64.13% |
| | **RoBERTa** | 51.71% | **70.05%** | 66.82% | 67.55% |
| `var_4_all_no_hash_no_elong` | **BERT** | 57.44% | 65.44% | 60.09% | 65.11% |
| | **RoBERTa** | 50.67% | **69.74%** | 66.99% | 67.61% |
| `var_5_all_lowercased` | **BERT** | 58.38% | 66.44% | 61.28% | 65.36% |
| | **RoBERTa** | 50.88% | **69.86%** | 65.34% | 67.55% |

> **Observation:** RoBERTa consistently outperformed BERT on LinearSVC, MLP, and XGBoost across all variations — often by 3–5%. The sole exception is Naive Bayes, where BERT embeddings scored higher because BERT's less-discriminative `[CLS]` representations happened to be more normally distributed, suiting NB's Gaussian assumption better.

---

### Phase 2: Fine-Tuned RoBERTa — Ablation Study Results

End-to-end fine-tuning of `roberta-base` with LLM-augmented synthetic training data across all 5 preprocessing variations:

| Variation | Accuracy | F1 Score | Misclassified (out of 252) |
|:---|:---:|:---:|:---:|
| `var_1_all_preprocessing` ⭐ | **97.62%** | **97.60%** | 6 |
| `var_2_normalising_casing_space` | 94.05% | 94.03% | 15 |
| `var_3_all_no_emoji_handling` | 96.43% | 96.42% | 9 |
| `var_4_all_no_hash_no_elong` | 97.62% | 97.62% | 6 |
| `var_5_all_lowercased` | 97.62% | 97.62% | 6 |

> ⭐ **Key Insight:** `var_1` (full preprocessing) and `var_4`/`var_5` share the same peak accuracy, but `var_2` (minimal cleaning) drops to 94.05% — confirming that even a powerful transformer benefits from structured preprocessing. The 3.57% gap between `var_1` and `var_2` represents **9 additional misclassifications**, largely on sarcastic/neutral comments.

---

### Fine-Tuning Error Analysis — Misclassification Patterns

#### Error Count by Variation

| Variation | Total Errors | Neutral→Positive | Neutral→Negative | Negative→Positive | Negative→Neutral | Positive→Negative |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `var_1_all_preprocessing` | 6 | 2 | 2 | 0 | 1 | 0 |
| `var_2_normalising_casing_space` | 15 | 2 | 7 | 2 | 1 | 2 |
| `var_3_all_no_emoji_handling` | 9 | 4 | 2 | 2 | 1 | 0 |
| `var_4_all_no_hash_no_elong` | 6 | 2 | 0 | 1 | 2 | 1 |
| `var_5_all_lowercased` | 6 | 2 | 0 | 1 | 2 | 0 |

#### Persistent Misclassifications (Errors shared across ≥ 2 variations)

| Comment | True Label | Most Common Error |
|:---|:---:|:---:|
| *"The behind the scenes construction footage was a nice addition. Doesn't change my view of the challenge itself."* | Neutral | → Positive |
| *"marcus got very little screen time before the elimination. was there more footage that didn't make the cut"* | Neutral | → Negative |
| *"The interesting variable here is that by hour 50 with three people left the challenge should have been more intense..."* | Negative | → Neutral |
| *"priya being two sections away at hour 90 means she covered more of the maze in her final push than anyone else imo"* | Neutral | → Positive |
| *"Good for Priya but the whole challenge was so clearly edited to make her look like the hero from the start"* | Negative | → Positive |

> **Root Cause:** These persistent errors share a common trait — **hedged or compound sentiment**. The model struggles with comments that contain both a compliment and a qualifier (e.g., *"nice addition. Doesn't change my view"*), or sarcasm disguised as praise (*"Good for Priya but..."*). These represent genuinely ambiguous cases where even human annotators may disagree.

---

### Frozen Base Model Misclassification Breakdown (RoBERTa + LinearSVC)

From the full misclassification log (13,500 total errors across all embeddings × classifiers × variations):

| Classifier | Total Misclassifications |
|:---|:---:|
| Naive Bayes | 4,399 |
| MLP NeuralNet | 3,384 |
| XGBoost | 3,108 |
| LinearSVC | 2,609 |

> **LinearSVC** produced the fewest misclassifications overall, reinforcing its position as the strongest frozen-embedding classifier. The RoBERTa + LinearSVC combination specifically misclassified **71 comments** where `True_Label = Positive` but `Predicted_Label = Negative` — predominantly slang-heavy positive comments (e.g., *"not gonna lie tHE TWIST 🤯 no cap"*) where the frozen `[CLS]` token failed to capture colloquial enthusiasm without fine-tuning.

---

## 🚀 Technologies Used
* **Languages:** Python, C (Custom Tokenization Logic)
* **Libraries:** Hugging Face `transformers`, `sentence-transformers`, `scikit-learn`, `xgboost`, `pytorch`, `pandas`, `numpy`.
* **Hardware:** Dual T4 GPUs (for RoBERTa fine-tuning), Multi-core CPUs (for XGBoost/Ensemble histogram processing).