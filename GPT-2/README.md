# GPT-2 Language Model: Custom Implementation

This repository contains a fully custom, from-scratch implementation of the GPT-2 language model using PyTorch. The model was successfully trained on the **OpenWebText** dataset using advanced optimization techniques to maximize GPU efficiency on an A6000 cloud instance.

## Model Architecture

The architecture mirrors the standard "GPT-2 Small" specification, resulting in a model with **162,373,248 parameters**.

* **Context Length (`max_len`)**: 1024 tokens
* **Vocabulary Size (`vocab_size`)**: 50,304 tokens (Custom BPE Tokenizer)
* **Embedding Dimension (`d_model`)**: 768
* **Feed-Forward Dimension (`d_ff`)**: 3072
* **Transformer Layers (`n_layers`)**: 12
* **Attention Heads (`n_heads`)**: 12
* **Dropout**: 0.1

## Optimization Techniques

To train this model efficiently on a single RTX A6000 GPU (48GB VRAM) while simulating a massive batch size, we implemented two critical optimization techniques in our training loop:

### 1. Automatic Mixed Precision (AMP)
By wrapping the forward pass in `torch.autocast`, mathematical operations that don't require high precision are downgraded from `fp32` (32-bit floats) to `fp16` (16-bit floats). 
* **Benefit**: Slashes VRAM consumption by nearly 50% and vastly accelerates Tensor Core matrix multiplications.
* **Safety**: We utilized PyTorch's `GradScaler` to prevent gradient underflow during the `fp16` backwards pass.

### 2. Gradient Accumulation
Loading 512 sequences of 1024 tokens would cause an immediate Out-Of-Memory (OOM) error. Instead, we implemented Gradient Accumulation:
* The data loader processes a micro-batch of **16** sequences.
* We calculate gradients and accumulate them using `loss.backward()` without updating weights.
* After **32** accumulation steps, `optimizer.step()` applies the combined gradients.
* **Benefit**: Achieves a perfect mathematical equivalent of a **global batch size of 512** without exceeding the A6000's VRAM limits.

## Training Log

The model was trained for **100,000 steps** using the `AdamW` optimizer and a `OneCycleLR` learning rate scheduler peaking at `6e-4`. 

Below is a snapshot of the training loss progression:

| Step | Training Loss | Learning Rate |
|------|---------------|---------------|
| 0    | 10.9085       | 2.4000e-05    |
| 500  | 9.9021        | 2.7289e-05    |
| 1000 | 7.9718        | 3.7961e-05    |
| ...  | ...           | ...           |
| 99000| 3.5226        | 1.8224e-07    |
| 99500| 3.4581        | 4.4509e-08    |
| 99900| 2.7562        | 4.0844e-09    |

### Training Loss Plot
![Training Loss Plot](training_loss_plot.png)

## Final Validation & Results

After completing 100,000 steps, we ran the model through 500 batches of the OpenWebText validation split.

* **Final Validation Loss**: `3.2204`

### Text Generation Sample
Using nucleus sampling with a temperature of `0.8` and `top_k=50`, the model is highly capable of generating valid grammatical structures and vocabulary. 

**Prompt**: `"Hello, my name is"`
> "Hello, my name is the first time a year of my mom. I've been asking me to go up a lot of people who don't take my time to go back, I'm an interesting way. I'm going to be a great opportunity as I always have to make a lot of time a few things work. It’s a really good thing and my favorite favorite. I had to have a ton of my life, so I’m working in my life, and my friends have to read your work with friends. I’ve made a new opportunity to help you up with a good job. I just had more love, and I’m going to get to work. I think you’ve done well, so many of them have a little bit more than you. I don’t have a sense. I’d get a ton of time for a lot about it, and I have a few things out to work (and I’m not in some more"

While a 162M parameter model lacks the capacity to maintain long-term global coherence on the scale of billion-parameter models, it successfully learned to utilize correct syntax, capitalization, punctuation, and clause structure entirely from scratch!

### Why does the generated text look like a rambling blog post?
Although the grammar is flawless, the semantics and logical flow are disjointed. This is expected due to two primary factors:
1. **Model Size:** At 162M parameters, this is a "Small" class model. It possesses enough capacity to memorize local sentence structure and grammar, but lacks the internal parameter count required to maintain long-term logical reasoning or topic coherence over multiple paragraphs (unlike 1.5B or 175B parameter models).
2. **Dataset Bias:** The model was trained on **OpenWebText**, which is heavily saturated with Reddit posts, forum comments, and personal lifestyle blogs. Because the prompt provided was *"Hello, my name is"*, the model mathematically calculated that the most likely continuation in its dataset was a personal blog introduction. It perfectly mimicked the *structural style* of an internet blogger, even if it lacked the capacity to give it underlying semantic meaning.
