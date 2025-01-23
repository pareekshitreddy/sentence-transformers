import torch
from transformers import AutoTokenizer

from models import MultiTaskSentenceTransformer

def test_multi_task_model():
    sentences = [
        "John lives in New York.",
        "I love using Transformers for NLP!"
    ]
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Suppose 3 classes for sentence classification (positive/negative/neutral)
    # Suppose 5 labels for NER 
    multi_task_model = MultiTaskSentenceTransformer(model_name=model_name,
                                                    num_classes_taskA=3,
                                                    num_labels_taskB=5)

    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = multi_task_model(input_ids=encoded["input_ids"], 
                                   attention_mask=encoded["attention_mask"])
    
    print("Task A logits (Sentence Classification):", outputs["taskA_logits"].shape)
    print("Task B logits (NER - Token Classification):", outputs["taskB_logits"].shape)
    # e.g. 
    # Task A logits shape -> [batch_size, 3]
    # Task B logits shape -> [batch_size, sequence_length, 5]

if __name__ == "__main__":
    test_multi_task_model()
