# SKGF

![](https://img.shields.io/badge/version-1.0.0-blue)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Pytorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)](https://www.pytorch.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/eduzrh/SKGF/issues)

[English](README.md) | [简体中文](./README_zh_CN.md)

🚀 **欢迎来到 SKGF (Scientific Knowledge Graph Fusion) 的仓库！** 🎉🎉🎉

---

## 🏠 **Overview** 🔍

**Scientific Knowledge Graph Fusion (SKGF)** 是一个**创新且实用的任务**，旨在填补**科学知识图谱（KG）丰富化**中的关键空白。现有方法主要聚焦于从非结构化数据中提取知识或通过内部推理完成 KG，但其范围和质量有限。**SKGF** 通过系统性地利用**全面、高质量的通用知识图谱（GKGs）** 来补充**科学知识图谱（SKGs）**，从而提升其完整性和实用性。💡

### 🎯 **Core Challenges** ⚡

1. **High Ambiguity of Domain Relevance** 🤔

   * 难以判断 GKG 中的知识是否真正与目标科学领域相关。

2. **Granularity and Alignment Mismatch between Scientific and General Facts** 🔄

   * GKG 事实通常抽象且粗粒度；
   * SKGs 需要更具**上下文**、**细粒度**的表示，以适应科学领域场景。

### ✨ **Key Innovations** 🌟

#### 1. **Novel Task Definition** 📋

* 首次系统探索从 GKGs 中挖掘并整合相关事实到 SKGs 中。
* 为**科学知识图谱丰富化** 奠定基础方法论。

#### 2. **Self-Fusion Framework** 🔗

一个**渐进式模糊融合框架**，配备**熵驱动的自反馈机制**，包含两个核心阶段：

**Stage 1: Fuzzy Retriever** 🕵️‍♂️（熵最大化，保留高熵候选以防丢失潜在证据）

**Stage 2: Entropy-driven Progressive KG Fusion** 🚀（基于自反馈循环迭代降低关系不确定性，将候选提炼为确定性事实）

#### 3. **Comprehensive Benchmarking** 📊

* **4 个新基准数据集**：SKGF(W-Bio)、SKGF(W-Music)、SKGF(W-Mat)、SKGF(W-Plant)，涵盖**生命科学、物理科学、社会科学**。
* **21 representative benchmark configurations**：用于系统性能评估。
* 跨多样化场景的**广泛实验验证**。

### ⚡ **Key Advantages** 💪

* **Superior Performance**：比基线方法提升**高达 20.0% + **。
* **Scientific Knowledge Granularity Alignment**：有效处理科学知识与通用知识之间的粒度与对齐错配。
* **Robust Relevance Detection**：高级模糊感知机制，确保准确科学领域相关性判断。
* **Scalable Architecture**：适应各种科学知识图谱场景。

📈 通过广泛实验验证，**SKGF** 在**科学知识图谱丰富化** 中确立了新的**最先进性能**，相比 **22** 个最先进基线具有持续优势，为将通用知识整合到科学专业领域提供实用范式。

---

## 🏗 **Architecture** 🏗️

SKGF 的**核心架构**采用**熵驱动的渐进式自反馈框架**，分为两个主要阶段：**Fuzzy Retriever**（模糊检索）和 **Entropy-driven Progressive KG Fusion**（熵驱动的渐进式 KG 融合）。

* **Fuzzy Retriever** 🕵️‍♂️：从 GKG 中**模糊检索**相关实体和三元组，支持**语义**和**结构**感知。
* **Progressive KG Fusion** 🚀：通过**场景生成**和**一致性验证**逐步整合知识，实现**自反馈优化**。
* **完整细节**：详见论文第 3 节，包括交互流程和伪代码。🔍

---

## 🔨 **Main Dependencies** 🛠️

* **Python** >=3.7 (测试于 Python=3.8.10) 🐍
* **PyTorch** >=1.10.0 🔥
* **Transformers** >=4.20.0 🤖
* **Scipy** >=1.7.0 📊
* **Pandas** >=1.3.0 🐼
* **Tqdm** >=4.62.0 ⏳
* **Numpy** >=1.21.0 🔢
* **NetworkX** >=2.6.0 🌐

---

## 📦 **Installation** ⚙️

兼容 **Python 3**。🚀

1. **创建虚拟环境** (可选，但推荐避免依赖冲突)

   ```shell
   conda create -n SKGF python=3.8.10
   conda activate SKGF
   ```

2. **配置 OpenAI API** (可选，用于语义检索和场景生成) 🔑
   在 `main.py` 或 `.env` 文件中设置你的 `OPENAI_API_BASE` 和 `OPENAI_API_KEY`。示例：

   ```env
   OPENAI_API_KEY=your_key_here
   OPENAI_API_BASE=your_base_here
   ```

---




## ✨ **Datasets** 📁

### **DKGF(Y-I)** 🗺️ (时政危机)

### **DKGF(W-I)** 🌐 (时政危机)

### **SKGF(W-Bio)** 🧬 （生命科学）

### **SKGF(W-Music)** 🎵 （社会科学）

### **SKGF(W-Mat)** ⚛️ （物理科学）

### **SKGF(W-Plant)** 🌿 （生命科学）


### 🔗 下载链接

<div align="center">


[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/1G5WXDyvcqzEu-RuR0RuRnB8n24rhgW7-?usp=sharing)

</div>

> 🔐 密码：`skgf`

**准备步骤**：下载后解压至 `./dataset/` 目录，支持自定义数据集（只需匹配 JSON/CSV 格式：实体列表 + 三元组文件）。自定义科学领域？只需提供 SKG/GKG 子图！🔧

---

## 🔥 **Quick Start** ⚡

快速上手 **SKGF**，从克隆到运行只需 5 分钟！⏱️

1. **克隆仓库**

   ```bash
   git clone https://github.com/eduzrh/SKGF.git
   cd SKGF
   ```

2. **准备数据集**

   ```bash
   # 下载并解压数据集至 dataset/
   ```

   `DATASET_NAME` 可为 `W-I-S1`、`Y-I-S1`，`W-Bio`、`W-Music`、`W-Mat`、`W-Plant`，或自定义数据集（置于 `./dataset/`）。

3. **运行主实验**

   ```bash
   python main.py --data DATASET_NAME
   ```

   这将运行**完整 SKGF 流程**，输出融合后的三元组到 `./message_pool/output_triples_fusion.txt`。监控进度通过 Tqdm 条！📈

4. **查看结果**

   * **性能指标**：控制台打印实体/融合准确率、精确率、召回率和 F1 分数（e.g., Entity F1: 0.85）。
   * **时间/令牌消耗**：自动计算平均处理时间和 OpenAI 令牌使用（e.g., Avg Time: 2.3s/entity）。
   * **日志文件**：检查 `./logs/` 中的详细输出。🔍

---

## 🧪 **Benchmark Configurations** 📈

我们设计了 **21 个代表性基准配置**，用于**消融研究（Ablation Study）**，系统评估每个组件的影响。配置通过命令行标志控制（以 `--wo-` 前缀表示“移除”该组件）。具体可见论文第 4 节的表格。📋

### **消融分类** 🏷️

* **Fuzzy Retriever 系列** ：测试检索阶段子模块（如语义 vs. 结构感知）。
* **Progressive KG Fusion 系列**：测试融合阶段子模块（如场景生成 vs. 一致性检查）。
* **子模块消融**：细粒度变体，结合多个移除。

| **配置 ID**                              | **描述**          | **命令示例**                          |
| -------------------------------------- | --------------- | --------------------------------- |
| **C1: Full SKGF**                      | 完整框架（所有组件启用）。   | `python main.py --data DATASET_NAME` |
| **C2: w/o Fuzzy Retriever**            | 跳过检索，直接使用所有实体对。 | `--wo-fuzzy-retriever`            |
| **C3: w/o Line Graph Trans**           | 跳过元知识线图转换。      | `--wo-line-graph-trans`           |
| **C4: w/o Semantic Retrieval**         | 跳过语义模糊检索。       | `--wo-semantic-retrieval`         |
| **C5: w/o Structural Perception**      | 跳过结构模糊感知。       | `--wo-structural-perception`      |
| **C6: w/o Progressive Fusion**         | 跳过渐进融合，直接过滤三元组。 | `--wo-progressive-fusion`         |
| **C7: w/o On-Demand Integration**      | 跳过场景感知按需整合。     | `--wo-on-demand-integration`      |
| **C8: w/o Scene Generation**           | 跳过场景生成判断。       | `--wo-scene-generation`           |
| **C9: w/o Scene Graph Reconstruction** | 跳过场景图重构一致性检查。   | `--wo-scene-graph-reconstruction` |

**运行示例**：

```bash
# C2: 移除模糊检索，快速测试基线
python main.py --data W-Bio --wo-fuzzy-retriever
```

**结果分析**：每个配置输出控制台指标（**实体性能**：准确率/精确率/召回率/F1；**融合性能**：类似）。使用 `./fusion_eval.py` 离线计算更多细节，如 ROC 曲线。📊💥

---

## 🧑‍💻 **Advanced Usage: Ablation Studies** 🔬

**SKGF** 支持**全面消融实验**，通过标志灵活控制组件：🛠️

* `--wo-fuzzy-retriever`：移除**Fuzzy Retriever**整个阶段。
* `--wo-progressive-fusion`：移除**Progressive KG Fusion**，退化为简单过滤。
* 子模块：`--wo-line-graph-trans`、`--wo-semantic-retrieval` 等（仅当父模块启用时生效）。

**流程概述**（详见 `main.py` 中的 `run_full_process()` 函数）：

1. **提取相关实体/三元组**：基础数据准备，从 SKG/GKG 加载。📥
2. **Fuzzy Retriever**：线图转换 → 语义检索 → 结构感知 → 实体转换。🔍
3. **Progressive KG Fusion**：按需整合 → 过滤/排名三元组 → 场景生成 → 一致性验证 → **反馈循环**（迭代优化阈值）。🔄
4. **去重 & 评估**：计算**时间/令牌**消耗及性能指标，使用 NetworkX 可视化融合图。🌐

**自定义**：修改 `args` 以启用/禁用反馈 (`--feedback-loop`) 或调整阈值 (`--threshold 0.7`)。运行后，检查 `./message_pool/` 中的中间文件（如 `rank_temp_output_triples_fusion.txt`）进行调试。🐛

**提示**：对于大规模数据集，监控 OpenAI 令牌使用（例如，新增启用 `--dry-run` 预估）。⚠️ 建议使用 GPU 加速 PyTorch 模块（`--device cuda`）。

---

## 📊 **Evaluation Metrics** 📏

我们采用标准 KG 评估指标，确保**可比性和透明度**：📐

* **Entity Performance**：实体对齐的 **Acc/Prec/Rec/F1**（基于 Levenshtein 相似度阈值 0.8）。
* **Fusion Performance**：融合三元组的 **Acc/Prec/Rec/F1**（参考 `ref_triples_1_fusion` vs. `false_ref_triples_1_fusion`，使用 RDF 一致性检查）。
* **Efficiency**：平均时间（s/entity）、令牌消耗（tokens/call），以及整体可扩展性（O(n) 复杂度分析）。


运行 `./eval_full.py` 生成详细报告和图表！📈

---

## 🌍 **Contact Information** 📞

📢 有任何问题或反馈？欢迎联系我们！我们非常欣赏您的建议！🙌

* 📧 **Email**： [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
* 📝 **GitHub Issues**：技术问题请在 [GitHub 仓库](https://github.com/eduzrh/SKGF/issues) 创建 Issue。标签：`bug`、`enhancement` 或 `question`。

我们将在 **2-3 个工作日内** 回复所有问题。⏱️

---

## 📜 **License** ⚖️

[MIT License](LICENSE) - 保留版权声明。🆓

---

## **Happy Researching** 🌟

**Stay tuned for updates!** ⭐ **Star 这个仓库** 以关注我们的进展。让我们一起**淘金**知识图谱！⛏️

**Acknowledgments**：感谢所有贡献者和审稿人！❤️ 特别致谢 PyTorch 社区和 OpenAI API 支持。
