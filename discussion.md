# Discussion Questions

## 3.1 Which parts of the network to train vs. keep frozen?

When training a multi-task sentence transformer:

### Freezing the Transformer Backbone
- **Makes sense if** you already have a very strong pretrained model that:  
  1. Generalizes well  
  2. You have limited computational resources or limited data  
- In practice, you often see **partial or full fine-tuning** of the backbone, but there are scenarios (e.g., real-time systems) where you might want to freeze the backbone to speed training or ensure you don’t forget the pretrained knowledge.

### Freezing One Head While Training the Other
- Useful if **Task A** is already “well-trained” and you do not want to negatively impact its performance, but you want to continue training **Task B**.
- For instance, if Task A’s head is critical to your application and you only want to add a second task (Task B) without risking performance regressions on the first.
- Another scenario: If you suspect **catastrophic forgetting**, freezing the head for the stable task can help preserve that capability while focusing on the new task.

In general, many practitioners will **fully fine-tune the entire model** if they have enough data. However, in **low-data** scenarios for one of the tasks, you might freeze large portions of the network and only fine-tune the heads or only certain layers (like the top few Transformer layers).

---

## 3.2 When to implement a multi-task model vs. separate models?

### Multi-Task Model

**Pros**:
- Parameter sharing can lead to better generalization if tasks are related.
- Less memory usage at inference time (one backbone, multiple heads).
- Potentially faster inference if you only do one forward pass for multiple outputs.

**Cons**:
- If tasks differ significantly, they may interfere with each other (**negative transfer**).
- Architecture can become more complex to tune.
- Catastrophic forgetting or performance trade-offs might be introduced.

### Separate Models

**Pros**:
- Each model can specialize in its own task.
- Simpler training procedure no need to manage multi-task losses or sampling strategies.

**Cons**:
- Double the parameters and possibly double the inference time if you need both tasks at the same time.
- Harder to exploit synergies between tasks (you lose the benefits of shared knowledge).

Ultimately, if tasks are sufficiently related and you want to capitalize on shared language knowledge (like a single representation that’s beneficial for both tasks), a multi-task model can be beneficial. If tasks are very different (e.g., one is an image-based task, another is text-based) or if you suspect negative transfer, separate models might be safer.

---

## 3.3 Handling Data Imbalance (Task A has abundant data, Task B has limited data)

When training the multi-task model with an **imbalanced dataset** (e.g., Task A has 1,000,000 samples, Task B has only 5,000), you could:

1. **Use Weighted Sampling**  
   Adjust the sampling so that during training, you don’t always see Task A examples. For instance, alternate minibatches between Task A and Task B or set sampling probabilities that ensure Task B is not overshadowed.

2. **Loss Weighting**  
   Adjust the multi-task loss such that Task B has a higher weight relative to Task A, helping the model pay more attention to Task B.

3. **Transfer Learning / Fine-tuning**  
   - Train the backbone (and maybe Task A’s head) on Task A (large dataset).  
   - Freeze (or partially freeze) the backbone and train Task B’s head on Task B’s smaller dataset.  
   - Optionally, unfreeze certain layers if you suspect you can still gain from some fine-tuning with Task B data.

4. **Data Augmentation**  
   For the low-resource task, create synthetic data or use techniques like back-translation (for classification tasks) or other data augmentation strategies (for NER) to bolster Task B.

Balancing these strategies depends on how **related** the tasks are. The more related they are, the more you might benefit from a shared backbone that’s been well-trained on the high-resource task.

