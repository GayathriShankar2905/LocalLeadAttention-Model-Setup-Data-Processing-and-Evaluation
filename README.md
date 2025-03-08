# LocalLeadAttention-Model-Setup-Data-Processing-and-Evaluation

Overview
This repository implements the LocalLeadAttention model for ECG classification using the MIT-BIH Arrhythmia dataset. The model leverages self-attention mechanisms and LSTMs to enhance feature extraction from ECG signals.

Features
Preprocessing: Noise removal, segmentation, normalization
Handling Imbalanced Data: SMOTE for minority class augmentation
Model Architecture:
1D CNN for feature extraction
Self-attention mechanism for sequence focus
Bidirectional LSTM for temporal dependencies
Fully connected layers for classification
Evaluation Metrics: Accuracy, F1-score, AUC-ROC
Dataset
Source: MIT-BIH Arrhythmia Database (PhysioNet)
Sampling Rate: 360 Hz
Classes: Normal beats, arrhythmias (e.g., PVCs, supraventricular beats)
Installation
Clone the repository and install dependencies:

bash
Copy
Edit
git clone https://github.com/yourusername/LocalLeadAttention.git
cd LocalLeadAttention
pip install -r requirements.txt
Usage
Run the training script:

bash
Copy
Edit
python train.py
Results
Metric	Score
Accuracy	87.5%
F1 Score	85.3%
Precision	88.1%
Recall	82.7%
AUC-ROC	90.2%
License
MIT License
