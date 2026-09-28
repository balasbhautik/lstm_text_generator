import os
import pickle
import requests
from sklearn.model_selection import train_test_split
from tensorflow.keras .callbacks import EarlyStopping, ModelCheckpoint
from src.dataset import Textpreprocessor
from src.models import build_lstm_model


# Obtain Dataset (Shakespeare Text sample)
DATA_PATH = "data/input_txt.txt"
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

if not os.path.exists(DATA_PATH):
    url = "https://raw.githubusercontent.com/karpaty/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()[:100000]

# dataset processing
SEQ_LENGTH = 40
preprocessor = Textpreprocessor(sequence_length=SEQ_LENGTH, level='word')
X, y = preprocessor.prepare_sequences(raw_text)

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save tokenizer for inference
with open("models/tokenizer.pkl", "wb") as  f:
    pickle.dump(preprocessor.tokenizer, f)

# Model building
model = build_lstm_model(
    vocab_size=preprocessor.vocab_size,
    sequence_length=SEQ_LENGTH,
    embedding_dim=64,
    lstm_units=128,
    deeper=False # set to true for stacked LSTM Comparison
)

# callbacks setup
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('models/best_lstm_model.h5', monitor='val_loss', save_best_only=True)
]

# Model Traning
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs = 40,
    batch_size=64,
    callbacks=callbacks
)



