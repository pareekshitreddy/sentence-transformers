import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentenceTransformer(nn.Module):
    """
    A simple Sentence Transformer model that uses a pretrained transformer 
    (e.g. 'bert-base-uncased') as its backbone and produces fixed-length embeddings.
    """
    def __init__(self, model_name="bert-base-uncased", output_dim=768, pooling="cls"):
        super(SentenceTransformer, self).__init__()
        self.model_name = model_name
        self.pooling = pooling

        # Load pretrained transformer backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # For example, we can project 768 -> output_dim. 
        # If output_dim == 768, the projection is not strictly necessary.
        if output_dim != self.backbone.config.hidden_size:
            self.projection = nn.Linear(self.backbone.config.hidden_size, output_dim)
        else:
            self.projection = None

    def forward(self, input_ids, attention_mask):
        # Pass inputs through transformer
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Transformer outputs: last hidden state, pooler output, etc.
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output  # [CLS] token representation (for BERT-like models)
        
        if self.pooling == "cls":
            # Use the [CLS] representation
            sentence_embedding = pooler_output
        elif self.pooling == "mean":
            # Mean pooling of the last hidden state
            # Note: We have to take attention_mask into account to properly average over non-padded tokens
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
            sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask
        else:
            raise ValueError("Invalid pooling type. Choose from ['cls', 'mean'].")
        
        # Optional projection
        if self.projection is not None:
            sentence_embedding = self.projection(sentence_embedding)
        
        return sentence_embedding


class MultiTaskSentenceTransformer(nn.Module):
    """
    A multi-task model that shares a transformer backbone for encoding sentences.
    Task A: Sentence Classification (positive/negative/neutral)
    Task B: Named Entity Recognition (token-level classification) 
            
    """
    def __init__(self, model_name="bert-base-uncased", 
                 num_classes_taskA=3,  # e.g. 3 classes for classification
                 num_labels_taskB=5):  # e.g. 5 labels for NER
        super(MultiTaskSentenceTransformer, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        # ---------- Task A: Sentence Classification Head ----------
        # We'll use the pooler output ([CLS] token representation) from the backbone
        hidden_size = self.backbone.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes_taskA)
        )
        
        # ---------- Task B: NER (Token Classification) Head ----------
        # We'll apply a linear layer to the last hidden state for each token
        self.ner_head = nn.Linear(hidden_size, num_labels_taskB)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass that returns outputs for both tasks:
        1) sentence classification logits
        2) token-level classification logits
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # The last hidden state: [batch_size, seq_len, hidden_dim]
        last_hidden_state = outputs.last_hidden_state
        # The pooler output: [batch_size, hidden_dim] (for BERT-like models)
        pooler_output = outputs.pooler_output  # used for the sentence classification

        # Task A (Sentence Classification) -> shape [batch_size, num_classes_taskA]
        logits_taskA = self.classifier_head(pooler_output)

        # Task B (NER / Token Classification) -> shape [batch_size, seq_len, num_labels_taskB]
        logits_taskB = self.ner_head(last_hidden_state)

        return {
            "taskA_logits": logits_taskA,
            "taskB_logits": logits_taskB
        }
