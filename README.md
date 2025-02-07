# Neural Machine Translation (NMT)

## Overview
This project focuses on building a Neural Machine Translation (NMT) model for translating between Italian and English. The system is implemented using PyTorch and employs an attention mechanism to enhance translation accuracy. The model is trained on the OPUS100 dataset, a widely used multilingual dataset, and evaluated using the BLEU metric, which measures translation quality based on reference sentences.

## Setup
To run this project, install the required dependencies using the following command:
```bash
pip install datasets sacrebleu torch torchvision torchaudio matplotlib seaborn tqdm
```
Ensure that you have Python 3.x installed and a working Jupyter Notebook environment.

## Usage
1. Open the notebook in Jupyter:
```bash
jupyter notebook "Rida st125481 a3.ipynb"
```
2. Follow the steps in the notebook to preprocess the dataset, train the model, and evaluate its performance.
3. The model outputs translated sentences, which are compared with reference translations using the BLEU score.

## Model Details
- **Architecture:** Sequence-to-Sequence (Seq2Seq) with Attention Mechanism
- **Dataset:** OPUS100 (English-Italian translation dataset)
- **Libraries Used:** PyTorch, Hugging Face Datasets
- **Evaluation Metric:** BLEU Score, which assesses the similarity between model-generated and reference translations.

## Evaluation
The performance of the model is evaluated using the BLEU metric. Higher BLEU scores indicate better translation accuracy. The results help determine how well the model generalizes to unseen text.

## File Structure
```
├── Rida st125481 a3.ipynb  # Jupyter notebook with implementation
├── README.md               # Project documentation
```

## License
This project is developed for educational purposes only and is not intended for commercial use.

