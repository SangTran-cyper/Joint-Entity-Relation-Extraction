# Joint Entity-Relation Extraction using Transformers 🌌

### 📝 Project Description & Objective
This project implements a multi-task learning architecture designed to extract entities (NER) and their semantic relations (RE) simultaneously from raw text. By moving away from traditional sequential pipeline methods, this joint extraction approach optimizes context sharing and completely eliminates the error propagation problem.

---

### 🛠️ Technologies
- **Core Language:** `Python`
- **Deep Learning Frameworks:** `PyTorch`, `HuggingFace Transformers`
- **Model Backbone:** `bert-base-cased` (Pre-trained Transformer)
- **Evaluation & Utilities:** `Scikit-learn`, `Matplotlib`
- **Infrastructure:** `Kaggle Notebooks (NVIDIA Tesla T4 GPU)`

---

### ✨ Core Features
- **Joint Multi-head Architecture:** Runs a single forward pass through BERT to generate two parallel prediction heads: token-level sequence output for Named Entity Recognition (NER) and sentence-level pooled output `[CLS]` for Relation Extraction (RE).
- **BIO Tagging Pipeline:** Automatically encodes entity labels into standard BIO format (7 classes) dynamically mapped with `BertTokenizer`.
- **Intelligent Post-Processing Layer:** A safeguarding logic layer that automatically sanitizes model outputs during real-time inference, forcing the relation to `NA` if fewer than two entities are extracted, drastically reducing hallucination.

---

### ⚙️ The Process
1. **Data Analytics & Mapping:** Analyzed the English New York Times (NYT) journalism dataset (over 61,000 samples) and visualized the heavy data imbalance where the `NA` label overwhelmingly dominates.
2. **Tokenization Alignment:** Tokenized cased text and mapped entity char-level spans to token-level indices while preserving exact word boundaries.
3. **Multi-task Training:** Combined the losses from both sub-tasks ($Loss_{total} = Loss_{ent} + Loss_{rel}$) using `CrossEntropyLoss` and optimized the parameters using `AdamW` with a learning rate of $2 \times 10^{-5}$ over 5 epochs.
4. **Validation & Logging:** Evaluated performance metrics on the test set using both Macro and Micro metrics to gauge overall fairness and exact benchmark comparisons.

---

### 🧠 What I Learned
- **Multi-task Learning Optimization:** Mastered how to design a shared backbone representation that simultaneously satisfies two distinct NLP objectives without letting one task dominate the loss function.
- **Handling Extreme Data Imbalance:** Learned how model performance metrics (Macro vs. Micro F1-score) diverge when a dataset is severely skewed toward negative labels (`NA`), and how to properly benchmark model capability.
- **Real-world Inference Edge Cases:** Understood that raw model outputs often violate strict domain logic, and realized the importance of building rule-based post-processing guards for safe production deployment.

---

### 🔮 How Can I Improve the Project?
- **Advanced Loss Functions:** Integrate `Focal Loss` or class-balanced loss weights to better handle the rare relation categories that suffer from data scarcity.
- **Entity-pair Aware Embeddings:** Instead of relying solely on the general `[CLS]` token for relation extraction, extract and concatenate the specific latent vectors of the target entity pair before the classification layer.
- **Graph Convolutional Networks (GCN):** Incorporate a GCN layer on top of BERT to better model the dependency trees and spatial relationships, especially for handling overlapping entities (Overlapping Triples).

---

### 🏃 Running the Project
1. **Clone the repository:**
```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
