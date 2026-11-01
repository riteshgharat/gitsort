# GitSort

GitSort classifies GitHub issues into six categories using fine-tuned Transformer models:

- Bug
- Feature Request
- Documentation
- Question
- Performance
- Security

## System Architecture

```mermaid
flowchart TD
    A["Raw GitHub Issue (Title + Description)"] --> B["Preprocessing & Normalization"]
    B --> C["[CODE] Block Masking & Markdown Cleaning"]
    C --> D["Tokenization & Attention Masking (max_len = 256)"]

    D --> E1["DistilBERT (66.96M)"]
    D --> E2["BERT-base (109.49M)"]
    D --> E3["RoBERTa-base (124.65M)"]
    D --> E4["DeBERTa-v3-small (141.90M)"]

    subgraph TrainingVariants["4 Systematic Variants per Model"]
        V1["Variant A: Single Fine-Tune"]
        V2["Variant B: Stratified 5-Fold CV"]
        V3["Variant C: Bagging (3 Bags)"]
        V4["Variant D: Boosting (3 Rounds)"]
    end

    E1 -.-> TrainingVariants
    E2 -.-> TrainingVariants
    E3 -.-> TrainingVariants
    E4 -.-> TrainingVariants

    E1 --> F["Probability Vector P_distil"]
    E2 --> G["Probability Vector P_bert"]
    E3 --> H["Probability Vector P_roberta"]
    E4 --> I["Probability Vector P_deberta"]

    F & G & H & I --> J["Combined Soft-Voting Ensemble"]
    J --> K["Predicted Category (Argmax)\n[Bug | Feature | Docs | Question | Perf | Security]"]
```

## Results

The combined soft-voting ensemble achieved:

- 83.07% test accuracy
- 96.63% ROC-AUC

The best standalone model was RoBERTa-base with the boosting variant:

- 82.65% test accuracy
- 80.90% macro F1

## Setup

Requirements: Python 3.9-3.11.

```bash
git clone https://github.com/<your-username>/gitsort.git
cd gitsort
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in `notebooks/` to reproduce the experiments.

## Dataset

The `data/` directory contains 3,541 labeled GitHub issues:

- `train.csv`: training data
- `label_map.json`: label-to-ID mapping

Each record includes the repository, issue title, description, label, and shard.

## Repository Structure

```text
data/       Dataset and label mapping
notebooks/  Training and evaluation notebooks
README.md   Project documentation
```

## License

This project is licensed under the MIT License.
