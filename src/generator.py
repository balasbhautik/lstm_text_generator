import numpy as np
import  tensorflow as tf


def sample_with_temperature(preds, temperature=1.0):
    """
    Applies temperature scaling to output probability distribution.
    """
    preds = np.array(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, tokenizer, seed_text: str, num_tokens_to_generate: int, sequence_length: int, temperature=0.7):
    """
    Iteratively predicts next tokens to build generated text output.
    """
    index_to_word = {v: k for k , v in tokenizer.word_index.items()}
    result = seed_text

    for _ in range(num_tokens_to_generate):
        # Clean and tokenizer current sequence seed
        cleaned_seed = seed_text.lower()
        encoded = tokenizer.texts_to_sequences([cleaned_seed])[0]

        # Pad or truncate squences to model input length
        encoded = tf.keras.preprocessing.sequence.pad_sequences(
            [encoded], maxlen=sequence_length, padding='pre'
        )

        # predict probability distribution
        preds = model.predict(encoded, verbose=0)[0]

        # Sample index using temperature selection
        next_index = sample_with_temperature(preds, temperature)
        next_char = index_to_word.get(next_index, '')

        # Append token to seed string
        if tokenizer.char_level:
            seed_text += next_char
            result += next_char
        else:
            seed_text += " " + next_char
            result += " " + next_char

    return result