# Plagiarism Detection Model

## Overview
This project implements a binary classification model to detect plagiarism between text pairs using a fine-tuned transformer model. The system analyzes two input sentences and determines whether the second sentence is plagiarized from the first one.

## Key Features
- Uses a lightweight transformer model (SmolLM-135M) for efficient processing
- Implements mixed precision training for performance optimization
- Balances the dataset using oversampling techniques
- Achieves high accuracy (96%) with minimal training

## Dataset
The model is trained on the SNLI (Stanford Natural Language Inference) dataset, filtered to focus on binary classification (plagiarized/non-plagiarized). The dataset is processed to:
- Balance class distribution through oversampling
- Subsample to 50% size for efficiency
- Split into train (72%), validation (8%), and test (20%) sets

## Model Architecture
- Base model: HuggingFaceTB/SmolLM-135M
- Adaptation: Fine-tuned for sequence classification with 2 output classes
- Input processing: Text pairs combined with [SEP] token
- Max sequence length: 128 tokens

## Performance
After 2 epochs of training, the model achieves:
- Test accuracy: 96.0%
- Non-plagiarized precision: 0.97, recall: 0.95, F1-score: 0.96
- Plagiarized precision: 0.95, recall: 0.97, F1-score: 0.96

## Technical Implementation
- Mixed precision training with amp.GradScaler for efficiency
- AdamW optimizer with learning rate of 2e-5
- CrossEntropyLoss function
- Batch size of 32 with 4 worker threads
- CUDA acceleration when available

## Usage
The code is structured to:
1. Load and preprocess text data
2. Prepare custom dataset and dataloader classes
3. Initialize and configure the transformer model
4. Train the model with optimization techniques
5. Evaluate performance on validation and test sets

## Requirements
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- tqdm

## Future Improvements
- Experiment with different transformer architectures
- Implement early stopping based on validation metrics
- Add more text preprocessing techniques
- Explore domain adaptation for specific use cases
