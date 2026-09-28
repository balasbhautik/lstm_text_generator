import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout



def build_lstm_model(vocab_size: int, sequence_length: int, embedding_dim=128, lstm_units=256, deeper=False):
    """
    Builds  and compiles an LSTM neural network for text generation.
    Supports a 'deeper' flag for architectural  experimenatation.
    """

    model = Sequential()

    # Embedding Layar
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))

    if deeper:
        # Stacked LSTM Architecture 
        model.add(LSTM(lstm_units, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(lstm_units))
        model.add(Dropout(0.2))
    else:
        # single LSTM Layers
        model.add(LSTM(lstm_units))
        model.add(Dropout(0.2))

    # Dense ouotput Layers with softmax
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
    )

    return model
