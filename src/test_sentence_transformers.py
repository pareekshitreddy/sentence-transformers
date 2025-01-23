import torch
from transformers import AutoTokenizer

from models import SentenceTransformer

def test_sentence_transformer():
    sentences = [
        "Hello world!",
        "Sentence Transformers provide powerful sentence embeddings.",
        "Transformers are great for NLP tasks."
    ]
    
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence_transformer = SentenceTransformer(model_name=model_name, output_dim=768, pooling="cls")
    
    # Prepare input batches
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Forward pass
    with torch.no_grad():
        embeddings = sentence_transformer(input_ids=encoded["input_ids"], 
                                          attention_mask=encoded["attention_mask"])
    
    print("Embeddings shape:", embeddings.shape)
    print("Sample embedding for first sentence:", embeddings[0])

if __name__ == "__main__":
    test_sentence_transformer()
