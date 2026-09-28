Generative AI with LSTM: Text Generation Project

This repository contains a complete implementation of a Sequence-to-Sequence / Next-Token Text Generator built using Python and TensorFlow/Keras. The project uses a Long Short-Term Memory (LSTM) Recurrent Neural Network trained on public-domain text (Shakespeare) to learn language patterns and generate coherent sequences from seed inputs.

📌 Features

Flexible Preprocessing: Pipeline supporting both character-level and word-level tokenization.

LSTM Neural Network Architecture: Sequential model with Embedding, LSTM, Dropout (for regularization), and Dense Softmax output layers.

Robust Training Pipeline: Includes train/validation splitting, early stopping, and automatic checkpointing for best validation loss.

Temperature-Controlled Generation: Configurable sampling temperature to balance randomness vs. predictability in generated text.

Architectural Experimentation: Easily configurable for deeper stacked LSTM networks.

📁 Project Structure

lstm_text_generator/
│
├── data/
│   └── input_text.txt          # Raw training dataset (e.g., Shakespeare text)
│
├── models/
│   ├── best_lstm_model.h5      # Saved Keras model checkpoint
│   └── tokenizer.pkl           # Saved tokenizer object
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── dataset.py              # Data loading, cleaning, and sequence formatting
│   ├── model.py                # LSTM neural network architecture definitions
│   └── generator.py            # Iterative text generation logic with temperature sampling
│
├── requirements.txt            # Project dependencies
├── train.py                    # Training pipeline script
├── generate_sample.py          # Script to run text generation using seed inputs
└── README.md                   # Project documentation


🛠️ Installation & Setup

1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

2. Environment Setup

Clone or download this repository, then set up a virtual environment:

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

Install all necessary packages via requirements.txt:

pip install -r requirements.txt


🚀 Running the Project

Step 1: Model Training

Run train.py to download the dataset automatically, preprocess the text into input-output sequence pairs, build the LSTM model, and train it:

python train.py


Output: The best model weights will be saved to models/best_lstm_model.h5, and token index mappings will be stored in models/tokenizer.pkl.

Step 2: Generating Text

Run generate_sample.py to produce generated text outputs based on seed sequences:

python generate_sample.py


🔬 Model Architecture & Design

Embedding Layer: Maps high-dimensional token index vectors to dense embedding representations.

LSTM Layer(s): Captures sequential dependencies across time steps. Includes Dropout (0.2) to mitigate overfitting.

Dense Output Layer: Softmax layer spanning the vocabulary size to predict probabilities for the next token in sequence.

Loss Function & Optimizer

Loss: sparse_categorical_crossentropy

Optimizer: Adam (adaptive learning rates)

📊 Temperature Sampling Strategy

Text generation uses temperature scaling on the predicted logits before multinomial sampling:

Low Temperature (0.2 - 0.4): Produces structured, repetitive, and highly conservative output.

High Temperature (0.7 - 1.0): Produces creative, diverse, but potentially noisy text.

📝 Deliverables Summary

Dataset: Publicly sourced from Tiny Shakespeare corpus.

Codebase: Fully modularized in src/ with structured scripts (train.py, generate_sample.py).

Sample Outputs: Verified via seed prompt evaluation in generate_sample.py.
