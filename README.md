
#  MMLLM-LongContext: Efficient Long-Context Reasoning in Multimodal LLMs

A research-driven framework to handle **long-context multimodal reasoning** with reduced memory consumption using:

-  **Streaming LLM**
-  **Cache Merging (CaM)**
-  **Image Chunking with Descriptive Generation**

Tested on SOTA models like **LLaVA-1.5**, **MiniGPT-4**, and **LLaVA-OneVision**, using **MileBench**.

---

##  Overview

Multimodal LLMs (MMLLMs) like LLaVA and MiniGPT4 struggle with long-context tasks due to memory-heavy **KV cache storage**. Existing methods either retain all tokens (high memory) or discard old ones (loss of context).

Our solution:
-  Use **Streaming LLMs** to dynamically limit attention.
-  Apply **Cache Merging** to compress KV entries.
-  Use **Image Chunking** to summarize images into compact text for scalable reasoning.

---

##  Datasets

###  MileBench

**MileBench** is a comprehensive benchmark built for testing **long-context multimodal reasoning**. It simulates real-world tasks that require LLMs to understand and reason across large sequences of interleaved images and text.

-  **Samples**: 6,440
-  **Total Images**: 97,855
-  **Avg. Words per Sample**: 422
-  **Avg. Images per Sample**: 15.2 (ranging from 2–109)
-  **Tasks**:
  - **Temporal Multi-image Reasoning (T-n)**
  - **Semantic Multi-image Reasoning (S-n)**
  - **Needle-In-A-Haystack Retrieval (N-n)**
  - **Image Retrieval (I-n)**

Each task stresses different forms of visual-textual alignment and memory retention over long sequences.

---

##  Evaluation

To run experiments on various models and token management methods, use:


python evaluation/benchmark_runner.py \
  --model [llava_1.5 | llava_1.6 | llava_onevision | minigpt4] \
  --method [streaming_llm | cache_merging | image_chunking] \
  --dataset milebench

##  Model Variants Evaluated

The following models were evaluated under different long-context management strategies:

| Model Name         | Type       | Description                                                                 |
|--------------------|------------|-----------------------------------------------------------------------------|
| **LLaVA-1.5-7B**   | Multimodal | Base version using CLIP-ViT and Vicuna backend. Commonly used in VQA tasks.|
| **LLaVA-1.6-7B**   | Multimodal | Improved version with enhanced image encoder and optimized context usage.  |
| **MiniGPT-4**      | Multimodal | Based on Vicuna + vision encoder, fine-tuned for image-to-text alignment.  |
| **LLaVA-OneVision**| Multimodal | Advanced version supporting multiple image tokens per prompt effectively.  |

Each of these models was benchmarked using three memory-efficient techniques: **Streaming LLM**, **Cache Merging (CaM)**, and **Image Chunking**.

---

##  Results and Benchmarks

### MMCoQA Accuracy on Long-Context Multimodal Tasks

| Model             | Method              | Few Images | Medium Images | Notes                                 |
|------------------|---------------------|------------|----------------|---------------------------------------|
| LLaVA-1.5-7B      | Full KV Cache       | 0.427      | 0.038          | High memory, good on small contexts   |
| LLaVA-1.5-7B      | Streaming LLM       | 0.400      | 0.210          | Better balance between mem & perf     |
| MiniGPT-4         | Descriptive Gen     | 0.390      | 0.340          | Compact summaries improve results     |
| LLaVA-OneVision   | Streaming LLM       | 0.354      | 0.240          | Multi-image performance tested        |

---

### Memory Footprint Comparison

| Model             | Strategy         | GPU Memory Usage |
|------------------|------------------|------------------|
| LLaVA-1.5-7B      | Full KV Cache    | 121 GB           |
| LLaVA-1.5-7B      | Streaming LLM    | 29 GB            |
| MiniGPT-4         | Descriptive Gen  | 32 GB            |
| LLaVA-OneVision   | Streaming LLM    | 30 GB            |

> ⚠ Full KV caching is unsustainable beyond ~6–8 images. Streaming LLM provides up to 75% memory savings.

---

###  Key Takeaways

- **Streaming LLM** reduces memory drastically with slight performance tradeoff.
- **Descriptive generation** via image chunking helps handle up to 100+ images.
- **CaM** must be tuned carefully — sometimes worsens memory under dense visual input.
- **LLaVA-OneVision** and **MiniGPT-4** are more scalable for large visual contexts than earlier LLaVA versions.

---

##  Qualitative Example: Long-Context Visual QA

**Prompt Example (from MileBench):**

> "In the sixth week of Destination: Deep Space, in which city can you locate the green bronze statue of the woman holding the torch?"

**Context**:
- 24 interleaved images and 3 paragraphs of historical and geographic text

**Answer**:
-  "New York City" (refers to the Statue of Liberty)

Models using **Streaming LLM + Descriptive Generation** processed all image chunks and context in under 35 GB of memory, while Full KV failed with OOM.

---

###  Contributors
- Dishant Zaveri
- Saransh Agrawal
- Pavan Santosh
- Faizan Ali

