import pickle
from tensorflow.keras.models import load_model
from src.generator import generate_text

SEQ_LENGTH = 40

# load  model and tokenizer
model = load_model("models/best_lstm_model.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# different seed sample
seeds = [
    "to be or not to be",
    "shall i compare thee",
    "once more unto the breach"
]

print("--- Generated Text Deliverables ---\n")
for seed in seeds:
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        seed_text=seed,
        num_tokens_to_generate=150,
        sequence_length=SEQ_LENGTH,
        temperature=0.2
    )
print(f"seed input : {seed}")
print(f"Output: {output}\n" + "-"*40)
