# SKGF

![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
[![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/SKGF/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

🚀 **Welcome to the SKGF (Scientific Knowledge Graph Fusion) Repository!** 🎉🎉🎉

This repository contains the source code for Self-Fusion.

---

## 🏠 **Overview** 🔍

**Scientific Knowledge Graph Fusion (SKGF)** represents an **innovative and practical task** aimed at addressing a critical gap in **scientific knowledge graph (KG) enrichment**. Existing approaches predominantly emphasize knowledge extraction from unstructured data or internal KG reasoning, which suffer from limitations in scope and quality. **SKGF** systematically harnesses **comprehensive, high-quality general knowledge graphs (GKGs)** to augment **scientific knowledge graphs (SKGs)**, thereby enhancing their completeness and utility. 💡

### 🎯 **Core Challenges** ⚡

1. **High Ambiguity of Domain Relevance** 🤔

   Determining the true relevance of knowledge from GKGs to the target scientific domain poses significant challenges.

2. **Granularity and Alignment Mismatch between Scientific and General Facts** 🔄

   Facts in GKGs are typically abstract and coarse-grained;
   In contrast, SKGs necessitate more **contextual** and **fine-grained** representations to accommodate scientific scenarios.

### ✨ **Key Innovations** 🌟

#### 1. **Novel Task Definition** 📋

This work pioneers the systematic exploration of mining and integrating relevant facts from GKGs into SKGs, laying a foundational methodology for **scientific knowledge graph enrichment**.

#### 2. **Self-Fusion Framework** 🔗

A **progressive fuzzy fusion framework** equipped with an **entropy-driven self-feedback mechanism**, encompassing two core stages:

**Stage 1: Fuzzy Retriever** 🕵️‍♂️ *(entropy maximization, retaining high-entropy candidates to avoid premature loss of latent evidence)*

**Stage 2: Entropy-driven Progressive KG Fusion** 🚀 *(iteratively reduces relation uncertainty via the self-feedback loop, refining candidates into deterministic facts)*

#### 3. **Comprehensive Benchmarking** 📊

* **Four new benchmark datasets**: SKGF(W-Bio), SKGF(W-Music), SKGF(W-Mat), SKGF(W-Plant), spanning **life sciences, physical sciences, and social sciences**.
* **21 representative benchmark configurations** for systematic performance evaluation.
* Extensive experimental validation across diverse scenarios.

### ⚡ **Key Advantages** 💪

* **Superior Performance**: Achieves improvements of **up to 20.0% +** over baseline methods.
* **Scientific Knowledge Granularity Alignment**: Effectively addresses the granularity and alignment mismatch between scientific knowledge and general knowledge.
* **Robust Relevance Detection**: Employs advanced fuzzy perception mechanisms to ensure precise scientific-relevance judgments.
* **Scalable Architecture**: Adaptable to a wide array of scientific KG scenarios.

📈 Through extensive experimental validation, **SKGF** establishes new **state-of-the-art performance** in **scientific knowledge graph enrichment**, consistently outperforming **22** state-of-the-art baselines, offering a practical paradigm for integrating general knowledge into scientific specialized domains.

---

## 🏗 **Architecture** 🏗️

The core architecture of **SKGF** adopts an **entropy-driven progressive self-feedback framework**, delineated into two primary stages: **Fuzzy Retriever** (fuzzy retrieval) and **Entropy-driven Progressive KG Fusion** (entropy-driven progressive KG fusion).

* **Fuzzy Retriever** 🕵️‍♂️: Conducts **fuzzy retrieval** of pertinent entities and triples from the GKG, incorporating both **semantic** and **structural** perception.
* **Progressive KG Fusion** 🚀: Facilitates progressive knowledge integration via **scenario generation** and **consistency verification**, enabling **self-feedback optimization**.
* **Full Details**: Refer to Section 3 of the accompanying paper for detailed interaction flows and pseudocode. 🔍

---

## 🔨 **Main Dependencies** 🛠️

* **Python** >= 3.7 (tested on Python 3.8.10) 🐍
* **PyTorch** >= 1.10.0 🔥
* **Transformers** >= 4.20.0 🤖
* **SciPy** >= 1.7.0 📊
* **Pandas** >= 1.3.0 🐼
* **Tqdm** >= 4.62.0 ⏳
* **NumPy** >= 1.21.0 🔢
* **NetworkX** >= 2.6.0 🌐

---

## 📦 **Installation** ⚙️

Compatible with **Python 3**. 🚀

1. **Create a Virtual Environment** (optional, but recommended to mitigate dependency conflicts)

   ```shell
   conda create -n SKGF python=3.8.10
   conda activate SKGF
   ```

2. **Configure OpenAI API** (optional, for semantic retrieval and scenario generation) 🔑
   Specify your `OPENAI_API_BASE` and `OPENAI_API_KEY` in `main.py` or a `.env` file. Example:

   ```env
   OPENAI_API_KEY=your_key_here
   OPENAI_API_BASE=your_base_here
   ```

---

## ✨ **Datasets** 📁

### **DKGF(Y-I)** 🗺️ *(Political Crises)*

### **DKGF(W-I)** 🌐 *(Political Crises)*

### **SKGF(W-Bio)** 🧬 *(Life Sciences)*

### **SKGF(W-Music)** 🎵 *(Social Sciences)*

### **SKGF(W-Mat)** ⚛️ *(Physical Sciences)*

### **SKGF(W-Plant)** 🌿 *(Life Sciences)*

### 🔗 Download Link

<div align="center">

[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/1G5WXDyvcqzEu-RuR0RuRnB8n24rhgW7-?usp=sharing)

</div>

> 🔐 Password: `skgf`

**Preparation Steps**: Download and extract the datasets into the `./dataset/` directory. Support for custom datasets is provided (adhering to JSON/CSV format: entity lists + triple files). For bespoke scientific domains, furnish SKG/GKG subgraphs. 🔧

---

## 🔥 **Quick Start** ⚡

Initiate **SKGF** swiftly, from cloning to execution in merely 5 minutes! ⏱️

1. **Clone the Repository**

   ```bash
   git clone https://github.com/eduzrh/SKGF.git
   cd SKGF
   ```

2. **Prepare Datasets**

   ```bash
   # Download and unzip datasets to dataset/
   ```

   `DATASET_NAME` may denote `W-I-S1`, `Y-I-S1`, `W-Bio`, `W-Music`, `W-Mat`, `W-Plant`, or a custom dataset (positioned within `./dataset/`).

3. **Run the Main Experiment**

   ```bash
   python main.py --data DATASET_NAME
   ```

   This invokes the **complete SKGF pipeline**, generating fused triples in `./message_pool/output_triples_fusion.txt`. Progress is monitored via Tqdm progress bars! 📈

4. **View Results**

   * **Performance Metrics**: Console outputs entity/fusion accuracy, precision, recall, and F1 scores.
   * **Time/Token Consumption**: Automatically calculates average processing time and OpenAI token usage.
   * **Log Files**: Examine detailed outputs in `./logs/`. 🔍

---

## 🧪 **Benchmark Configurations** 📈

We have devised **21 representative benchmark configurations** for **ablation studies**, enabling systematic assessment of each component's impact. Configurations are governed by command-line flags (prefixed with `--wo-` to denote "without" the component). Refer to Table 4 in Section 4 of the paper for specifics. 📋

### **Ablation Categories** 🏷️

* **Fuzzy Retriever Series**: Evaluates sub-modules in the retrieval stage (e.g., semantic versus structural perception).
* **Progressive KG Fusion Series**: Assesses sub-modules in the fusion stage (e.g., scenario generation versus consistency checking).
* **Sub-Module Ablations**: Fine-grained variants integrating multiple removals.

| **Configuration ID**                   | **Description**                                         | **Command Example**               |
| -------------------------------------- | ------------------------------------------------------- | --------------------------------- |
| **C1: Full SKGF**                      | Full framework (all components enabled).                | `python main.py --data W-Bio`     |
| **C2: w/o Fuzzy Retriever**            | Skip retrieval; utilize all entity pairs directly.      | `--wo-fuzzy-retriever`            |
| **C3: w/o Line Graph Trans**           | Skip meta-knowledge line graph transformation.          | `--wo-line-graph-trans`           |
| **C4: w/o Semantic Retrieval**         | Skip semantic fuzzy retrieval.                          | `--wo-semantic-retrieval`         |
| **C5: w/o Structural Perception**      | Skip structural fuzzy perception.                       | `--wo-structural-perception`      |
| **C6: w/o Progressive Fusion**         | Skip progressive fusion; apply direct triple filtering. | `--wo-progressive-fusion`         |
| **C7: w/o On-Demand Integration**      | Skip scenario-aware on-demand integration.              | `--wo-on-demand-integration`      |
| **C8: w/o Scene Generation**           | Skip scene generation judgment.                         | `--wo-scene-generation`           |
| **C9: w/o Scene Graph Reconstruction** | Skip scene graph reconstruction consistency check.      | `--wo-scene-graph-reconstruction` |

**Run Example**:

```bash
# C2: Remove fuzzy retrieval for rapid baseline testing
python main.py --data W-Bio --wo-fuzzy-retriever
```

**Result Analysis**: Each configuration yields console metrics (**entity performance**: accuracy/precision/recall/F1; **fusion performance**: analogous). Employ `./fusion_eval.py` for offline computation of supplementary details, such as ROC curves. 📊💥

---

## 🧑‍💻 **Advanced Usage: Ablation Studies** 🔬

**SKGF** facilitates **comprehensive ablation experiments** via flexible flag-based component control: 🛠️

* `--wo-fuzzy-retriever`: Eliminates the entire **Fuzzy Retriever** stage.
* `--wo-progressive-fusion`: Omits **Progressive KG Fusion**, reverting to rudimentary filtering.
* Sub-modules: `--wo-line-graph-trans`, `--wo-semantic-retrieval`, etc. (operative solely when parent modules are active).

**Workflow Overview** (elaborated in the `run_full_process()` function within `main.py`):

1. **Extract Relevant Entities/Triples**: Fundamental data preparation, loading from SKG/GKG. 📥
2. **Fuzzy Retriever**: Line graph transformation → semantic retrieval → structural perception → entity conversion. 🔍
3. **Progressive KG Fusion**: On-demand integration → triple filtering/ranking → scene generation → consistency verification → **feedback loop** (iterative threshold optimization). 🔄
4. **Deduplication & Evaluation**: Quantifies **time/token** consumption and performance metrics; visualizes fused graphs employing NetworkX. 🌐

**Customization**: Alter `args` to activate/deactivate feedback (`--feedback-loop`) or fine-tune thresholds (`--threshold 0.7`). Post-execution, scrutinize intermediate files in `./message_pool/` (e.g., `rank_temp_output_triples_fusion.txt`) for debugging. 🐛

**Tips**: For large-scale datasets, oversee OpenAI token utilization (activate `--dry-run` for estimation). ⚠️ GPU acceleration is advised for PyTorch modules (`--device cuda`).

---

## 📚 **Paper ↔ Code Correspondence** 🔗

This section maps the paper's methodology (Entropy-driven Progressive Self-Feedback for SKGF) directly onto the source modules so reviewers can navigate the codebase.

### Stage 1: Fuzzy Retriever

| Paper section | Module(s) |
|---|---|
| §Overview (Stage 1) | `main.py` (data preparation block, lines ≈ 27–110) |
| §Meta-knowledge Line Graph Transformation (Def. 3) | `fuzzy_retriever/line_trans.py::generate_line_triples_and_names` |
| §Semantic Fuzzy Retrieval (Eq. C_node, C_graph) | `fuzzy_retriever/semantic_rag.py::semantic_rag_all` (FAISS + OpenAI Embeddings) |
| §Structural Fuzzy Perception (Eq. δ_struct, δ_meta, C^g_meta) | `fuzzy_retriever/struc.py::structure_similarity_filter` (MCS + Edit-Distance variants) and `fuzzy_retriever/struct_enhance_1.py` (deep structural embedding) |

### Stage 2: Entropy-driven Progressive KG Fusion

| Paper section | Module(s) |
|---|---|
| §Fusion / Eq. objective (entropy minimization) | `main.py` (cycle-consistency block, lines ≈ 117–360) |
| §Scene-aware On-demand Integration (Eq. F_new) | `progressive_kg_fusion/on_demand_integration.py::integration` + `progressive_kg_fusion/logprob.py::filter_facts_by_tau` |
| §Entropy-driven Validity via Token Probability | `progressive_kg_fusion/logprob.py::logprob_yes_no` (softmax over binary vocabulary) |
| §Fusion Scene Generation (Prompt 2, Eq. S_desc) | `progressive_kg_fusion/scene_generation.py::scene_generate` |
| §Fusion Scene Graph Reconstruction (Prompt 3, Eq. H_cycle) | `progressive_kg_fusion/scene_graph_reconstruction.py::reconstruct_facts` |
| §Cycle Consistency & Negative Constraint (Algorithm 1 step 2.4) | `main.py` cycle block (`τ ← τ + δ`, `I_neg ← I_neg ∪ F_mismatch`, convergence on `F_mismatch == ∅`) |
| §Generation Entropy H_gen | `progressive_kg_fusion/entropy.py::sequence_entropy_from_top_logprobs` |
| §Cycle-consistency Entropy H_cycle (Eq. eq:cycle_entropy) | `progressive_kg_fusion/entropy.py::cycle_entropy` |

### Algorithm 1 Hyperparameters

Three new CLI flags expose the paper's entropy-driven loop:

| Flag | Default | Paper symbol |
|---|---|---|
| `--tau-init` | 0.5 | Initial entropy bound τ for F_new (Eq. F_new) |
| `--tau-delta` | 0.05 | Per-iteration increment of τ (Algorithm 1 step 2.4) |
| `--max-feedback-iterations` | 3 | Maximum cycle-consistency iterations |

### Backward Compatibility

The legacy entry points `judge_quadruple_facts` and `verify_fact_consistency` are preserved as thin wrappers around the original implementations. The original `verify_fact_consistency` had a duplicate definition in the source; that duplicate is preserved as `_legacy_verify_fact_consistency_duplicate` to avoid behavioural drift. Existing ablation flags (`--wo-*`) keep their original semantics:
- `--wo-scene-generation` → `F_recon := F_new` (cycle collapses)
- `--wo-scene-graph-reconstruction` → `F_valid := F_new` (no cycle check)

---

## 📊 **Evaluation Metrics** 📏

We utilize standard KG evaluation metrics to guarantee **comparability and transparency**: 📐

* **Entity Performance**: Alignment accuracy, precision, recall, and F1.
* **Fusion Performance**: Triple fusion accuracy, precision, recall, and F1 (benchmarking against `ref_triples_1_fusion` versus `false_ref_triples_1_fusion`, via RDF consistency checks).
* **Efficiency**: Average time (s/entity), token consumption (tokens/call), and overall scalability (O(n) complexity analysis).

Execute `./eval_full.py` to produce comprehensive reports and visualizations! 📈

---

## 🌍 **Contact Information** 📞

📢 For inquiries or feedback, we welcome your contact. Your suggestions are greatly valued! 🙌

* 📧 **Email**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
* 📝 **GitHub Issues**: For technical concerns, initiate an Issue in the [GitHub repository](https://github.com/eduzrh/SKGF/issues). Labels: `bug`, `enhancement`, or `question`.

Responses to all inquiries are targeted within **2-3 business days**. ⏱️

---

## 📜 **License** ⚖️

[MIT License](LICENSE) - Copyright notices preserved. 🆓

---

## **Happy Researching** 🌟

**Stay tuned for updates!** ⭐ **Star this repository** to track our advancements. Let us collectively **fuse** scientific knowledge graphs! 🔬

**Acknowledgments**: Profound gratitude to all contributors and reviewers! ❤️ Special acknowledgments to the PyTorch community and OpenAI API support.
