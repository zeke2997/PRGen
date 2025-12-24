# Code and Models for "[PRGen]"

This is the official repository for our paper, "[PRGen]".


# RELATED WORK

## Generic Traffic Representation & Analysis
The adaptation of Transformer architectures to network traffic has demonstrated remarkable efficacy in capturing complex dependencies within byte streams. NetGPT  adapts the GPT-2 architecture to traffic, employing a general hexadecimal encoding to unify diverse patterns into a text-like format for autoregressive generation. Similarly, TrafficLLM  leverages Large Language Models (LLMs) like Llama-2, introducing a traffic-domain tokenization strategy and a dual-stage tuning pipeline to align natural language instructions with raw traffic data. While primarily an encoder-based representation learner, TrafficFormer  aligns with the generative paradigm through its use of Masked Burst Modeling (MBM) as a pre-training objective. By reconstructing masked intervals within traffic "bursts," it learns robust structural embeddings from unlabeled data. However, these approaches generally treat traffic as a linear sequence or rely on standard NLP tokenization (BPE/WordPiece), which proves suboptimal for the high-entropy, rigid structural constraints of proprietary binary streams.

## Proprietary Protocol Traffic Generation
Research targeting proprietary or industrial protocols is sparse due to the lack of public specifications. The most relevant predecessor, PNetGPT, targets Industrial IoT (IIoT) protocols by mapping internal API function calls to network payloads. It employs an Encoder-Decoder architecture with a regex-based tokenization scheme designed to preserve numerical semantics (e.g., floating-point coordinates). However, PNetGPT operates under a "white-box" or "grey-box" assumption, necessitating access to the host software's internal API logs (function names and parameters) as conditioning inputs. This dependency renders it unsuitable for black-box security auditing or active scanning scenarios where internal device states are inaccessible.

## Positioning of PRGen
PRGen  distinguishes itself by targeting the challenging black-box generation of proprietary IoT probe responses, conditioned solely on external interaction context (e.g., probe parameters, device fingerprints) rather than internal API logs. Unlike PNetGPT's reliance on heuristic regex rules or TrafficLLM's general BPE, PRGen introduces Stripe-Aware Conditional Tokenization (SACT). SACT is a data-driven approach that utilizes entropy profiles to dynamically segment payloads into rigid "stripes" (static structures) and variable fields, preserving structural integrity without prior knowledge. Furthermore, we propose Entropy-Guided Masking (EGM), a pre-training strategy that strategically biases learning towards high-entropy regions, contrasting with the uniform masking or next-token prediction used in prior work. This positioning highlights PRGen's unique capability to synthesize valid binary payloads for unknown protocols in zero-knowledge environments.

## Table 1. Comparison of PRGen with State-of-the-Art Transformer-based Traffic Models

| Model         | Target Domain                        | Input Conditioning                        | Tokenization Strategy               | Training Strategy                                      | Generation Granularity |
|---------------|--------------------------------------|-------------------------------------------|--------------------------------------|--------------------------------------------------------|------------------------|
| **NetGPT**        | General / Encrypted Traffic          | Task Prompts                              | Hex + WordPiece                      | Autoregressive (CLM)                                   | Flow / Packet          |
| **TrafficLLM**    | Generic Malware / Web Attacks        | Natural Language Instructions             | Traffic-Domain BPE                   | Dual-Stage (Instruction + Traffic)                     | Flow / Packet          |
| **TrafficFormer** | General / Encrypted Classification     | Unlabeled Bursts                          | Bigram + BPE                         | Masked Burst Modeling (MBM) & SODF                     | Burst / Flow             |
| **PNetGPT**       | Proprietary Industrial Protocols     | API Function Names & Params               | Hex-split + Special Tokens           | Masked (MLM) & Autoregressive (ALM)                    | Payload Byte Stream      |
| **PRGen (Ours)**  | Proprietary IoT Probe-Response       | Interaction Context (Probe & Device Attr) | SACT (Entropy-Profile Driven)      | EGM (Entropy-Guided Masking)                           | Payload Byte Stream      |
