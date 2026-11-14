# Sentence Transformers and Multi-Task Learning

A BERT-based sentence encoder with two task heads trained jointly, written in PyTorch with Hugging Face Transformers.

The design choices are discussed in [the report](Sentence%20Transformers%20and%20Multi-Task%20Learning%20REPORT.pdf).

## Model

`MultiTaskSentenceTransformer` in `sentencetransformer.py`:

- **Encoder:** `bert-base-uncased`; the pooled CLS output is projected to a 256-dimensional, L2-normalised sentence embedding shared by both tasks.
- **Task A, sentence classification:** a linear head with two classes (positive, negative), trained with cross-entropy.
- **Task B, sentiment score:** a linear regression head with one output.
- **Training:** the two losses are summed and optimised together, so the shared embedding learns from both tasks.

## Data

`positive.txt` and `negative.txt` hold one sentence per line. The script labels, shuffles and splits them 80/20 into train and test sets.

## Run

```sh
pip install torch transformers scikit-learn
python sentencetransformer.py
```

The script trains the model and prints classification accuracy on the test split. It was exported from a Colab notebook, so it runs top to bottom as a single script.
