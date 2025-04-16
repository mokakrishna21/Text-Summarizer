#  BART Summarizer

This project fine-tunes **Facebook’s BART (base)** model for abstractive summarization using the **SAMSum** dataset. Powered by **PyTorch Lightning**, **Hugging Face Transformers**, and **ROUGE** metrics, it delivers precise summaries of conversational text.

---

## 🚀 Features

- 🧾 **Trains on SAMSum**: Realistic dialogue summaries  
- 🤖 **Fine-tunes BART-base** from HuggingFace  
- ⚡ **Fast Training** with PyTorch Lightning  
- 📊 **ROUGE Evaluation** for performance check  
- 🧪 **16-bit Precision Training** enabled  
- 💻 **CPU/GPU Compatible**

---

## 📚 Dataset

The **SAMSum** dataset contains multi-turn dialogues and their summaries.

Format:

| dialogue | summary |
|---------|---------|
| "Hi, how are you?" ... | "Friends catch up." |

Ensure the dataset file `samsum-train.csv` exists in the root directory.

---

## 🛠️ Getting Started

### 1. Requirements

- Python 3.10+
- NVIDIA GPU recommended (or fallback to CPU)

### 2. Installation

```bash
pip install transformers torch scikit-learn pandas pytorch-lightning datasets rouge-score evaluate
```

---

### 3. Configuration

Model & training config:

```python
MODEL_NAME = "facebook/bart-base"
MAX_SOURCE_LEN = 384
MAX_TARGET_LEN = 96
BATCH_SIZE = 8  # Reduce for CPU
MAX_EPOCHS = 1
PRECISION = 16  # Use 32 for CPU
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
```

---

### 4. Launch Training

```python
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    precision=PRECISION,
    accumulate_grad_batches=1,
    enable_progress_bar=True,
    accelerator="auto",
    devices="auto",
)
trainer.fit(model, train_loader, val_loader)
```

---

## ✨ Inference Guide

Use the function below to generate summaries:

```python
def summarize(text, model, tokenizer):
    inputs = tokenizer(
        text,
        max_length=MAX_SOURCE_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(model.device)

    summary_ids = model.model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=MAX_TARGET_LEN,
        num_beams=NUM_BEAMS,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## 📈 Sample Output

**Dialogue:**

> Anna: Hey! Just got back from the dentist.

> Mark: Oh no, how did it go?

> Anna: Not too bad, just a cavity. He filled it right away.

> Mark: That’s good. Were you in pain before?

> Anna: Yeah, a little. It started hurting last weekend.

> Mark: Oof. At least it’s sorted now.

> Anna: Yep, he said I need to floss more though

> Mark: Classic dentist advice

> Anna: Haha, yeah. Anyway, want to grab coffee later?

> Mark: Sure, let’s meet at 5 at the usual spot?

**Generated Summary:**

> Anna had a cavity filled and asked Mark to meet for coffee.

**Reference Summary:**

> Anna told Mark she had a cavity filled at the dentist and suggested meeting for coffee later.

**ROUGE Score:**

```json
{
  "rouge1": 0.5238, #good word overlap
  "rouge2": 0.1500, #fair phrase-level fluency
  "rougeL": 0.3810, #decent structural alignment
  "rougeLsum": 0.3810 #Higher is better, but I trained for 1 epoch max train for 3-4 for better results
}
```

---

## ⚙️ Customization

- **Change Model:**
```python
MODEL_NAME = "facebook/bart-large"  # Try larger versions
```

- **Adjust Training Parameters:**
```python
MAX_EPOCHS = 3
BATCH_SIZE = 4  # Adjust based on memory
```

- **Change Beam Search:**
```python
NUM_BEAMS = 5
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|--------|-----|
| ❌ CUDA Error | Reduce batch size or switch to CPU |
| ⛔ KeyError in CSV | Ensure correct headers in dataset |
| 📉 Low ROUGE | Train for more epochs or tune hyperparameters |
| 🚫 Tokenizer Errors | Match tokenizer with BART model version |

---

## 🤝 Contribution & Contact

Have feedback or want to contribute?  
📧 Email: [mokakrishna212@gmail.com](mailto:mokakrishna212@gmail.com)
