# my-kutti-model 🚀

An exploration of the transformer architecture to train a language model from scratch with available consumer infrastructure. Used a coding agent to create the training, inference, and benchmarking code. This exploration is a step in understanding the transformer architecture a tad better. 

**my-kutti-model** is a 1.97M-parameter open language model trained in three stages on a consumer laptop GPU.

*“Kutti” (குட்டி) — Tamil for small / little.*

## Overview

`my-kutti-model` is an experimental autoregressive language model designed to demonstrate full-lifecycle SLM (Small Language Model) development:
- **Architecture**: Pre-LayerNorm Transformer decoder (6 layers, 4 attention heads, $d_{\text{model}} = 160$)
- **Vocabulary**: 512-token Byte-Pair Encoding (BPE)
- **Three-stage Training Pipeline**:
  1. **Pre-training**: 74M token slice of TinyStories corpus
  2. **Mid-training**: 2,500 curated factual and narrative samples
  3. **Supervised Fine-Tuning (SFT)**: 100 instruction-response pairs with token-level loss masking
- **Deployment**: Exported to ONNX INT8 (2.3 MB) running 320–410 tokens/sec on CPU without GPU.

For full technical specifications, benchmark results, and methodology, see [MyKuttiModel_Whitepaper.md](MyKuttiModel_Whitepaper.md).

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Inference
```bash
python inference/run_inference.py
```

### 3. Evaluation
```bash
python evaluation/run_eval.py
```
