# EEGMoE: A Domain-Decoupled Mixture-of-Experts Model for Self-Supervised EEG Representation Learning

## Comprehensive Paper Analysis

**Paper Authors:** Xuange Gao, Danli Wang, and Yanyan Zhao  
**Affiliation:** Chinese Academy of Sciences & University of Chinese Academy of Sciences  
**Published in:** IEEE Transactions on Neural Networks and Learning Systems (2026)  
**DOI:** 10.1109/TNNLS.2026.3652277

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Key Contributions](#key-contributions)
4. [Foundational Concepts](#foundational-concepts)
5. [Methodology](#methodology)
6. [Experimental Setup](#experimental-setup)
7. [Results and Analysis](#results-and-analysis)
8. [Ablation Studies](#ablation-studies)
9. [Visualization and Interpretation](#visualization-and-interpretation)
10. [Critical Analysis and Future Directions](#critical-analysis-and-future-directions)
11. [References and Citations](#references-and-citations)

---

## Executive Summary

This paper introduces **EEGMoE** (EEG Mixture of Experts), a novel self-supervised pretraining model for EEG representation learning that addresses a critical limitation in existing EEG deep learning approaches. The key innovation lies in **domain decoupling** - the model simultaneously learns both domain-shared representations (commonalities across different EEG tasks/datasets/subjects) and domain-specific representations (unique characteristics of each domain) through a specialized Mixture-of-Experts (MoE) architecture.

### The Core Problem

Traditional EEG deep learning models suffer from three major limitations:

1. **Task Specialization:** Models are typically tailored for specific tasks, datasets, or even individual subjects, severely limiting their generalizability [15-19]
2. **Undecoupled Pretraining:** Recent large-scale pretraining approaches unify data formats but fail to decouple representations across domains, losing valuable domain-specific information
3. **Gradient Conflicts:** Different EEG tasks drive model parameters to optimize in conflicting directions during training (visualized in Figure 1 of the paper)

### The EEGMoE Solution

EEGMoE introduces a **Transformer-based domain-decoupled encoder** featuring a **Specific and Shared MoE (SSMoE) block** with:

- **Specific Expert Group:** Uses Top-K routing to dynamically select the most appropriate K experts for each input token, learning fine-grained domain-specific representations
- **Shared Expert Group:** Employs soft routing, leveraging all experts to learn domain-shared representations from every token

---

## Background and Motivation

### Understanding EEG Signals

**Electroencephalography (EEG)** is a physiological signal that directly measures brain activities through electrical activity recorded from electrodes placed on the scalp [1, 2]. EEG has proven indispensable in numerous neurological applications including:

- **Emotion Recognition (ER):** Identifying emotional states from brain activity patterns [3]
- **Motor Imagery (MI) Classification:** Detecting imagined movements for brain-computer interfaces [4]
- **Mental Workload Detection:** Assessing cognitive load during tasks [5]

### The Heterogeneity Challenge

EEG data exhibits significant heterogeneity across multiple dimensions:

1. **Task-Level Differences:** Different EEG tasks inherently differ in their:
   - Neural correlates (brain regions involved)
   - Dominant frequency bands (δ, θ, α, β, γ rhythms)
   - Temporal dynamics (time-varying patterns) [27-30]

2. **Subject-Level Variability:** EEG signals vary significantly across individuals due to:
   - Anatomical differences (skull thickness, brain structure)
   - Cognitive strategies
   - Physiological states [18, 24-26]

3. **Dataset-Level Variations:** Different recording setups introduce variability in:
   - Number and placement of electrodes (montage differences)
   - Sampling rates
   - Experimental protocols

### Why Current Approaches Fall Short

#### Task-Specific Training Paradigm

Traditional approaches train separate models for each task/dataset combination:

- **Handcrafted Features + Deep Learning:** Methods using differential entropy or power spectral density features combined with CNNs or RNNs [6-8]
- **Brain Connectivity Analysis:** Subject-dependent approaches for mental workload detection [9, 10]
- **End-to-End Deep Learning:** CNN, GNN, or Transformer backbones trained on specific datasets [11-16]

**Limitation:** While effective for their designated tasks, these models cannot transfer knowledge across domains [15-19].

#### Undecoupled Pretraining Paradigm

Recent self-supervised pretraining approaches attempt to learn unified representations:

- **MMM:** Multidimensional position encoding with multistage pretraining [21]
- **LaBraM:** Vector-quantized neural spectrum prediction for unified representations [22]
- **BIOT:** Flexible biosignal encoder with unified tokenization [46]
- **DMAE-EEG:** Masked autoencoder with neuroscientific priors [23]

**Limitation:** These approaches process all signals indiscriminately, overlooking potential discrepancies and conflicts among different tasks. As shown in Figure 1 of the paper, different tasks drive model parameters to optimize in distinct directions, causing gradient conflicts [31].

### The Mixture-of-Experts Opportunity

**Mixture of Experts (MoE)** architectures [50] employ multiple expert modules that adaptively activate specialized subnetworks for different inputs. Key advantages:

1. **Dynamic Routing:** Data is routed to task- or domain-specific experts
2. **Parameter Efficiency:** Only a subset of parameters is activated per input
3. **Task Adaptability:** Different experts can specialize in different subtasks

MoE has been successfully applied in:
- **Natural Language Processing (NLP):** [36, 37, 51, 53]
- **Computer Vision (CV):** [38, 39, 52]

**Existing EEG MoE Approaches and Their Limitations:**

- **IDMMoE:** Identity-based multi-gate MoE for emotion recognition - allocates subspace per subject but lacks flexibility [40]
- **Seizure-MoE:** Uses weighted-sum approach for seizure subtype classification - undermines MoE's selective activation advantage [41]
- **MoGE:** Mixture-of-graph-experts for cross-subject ER - remains a supervised feature extractor for a single task [42]
- **EEGMamba:** Incorporates MoE for multitask classification - primarily for multitask compatibility, not domain decoupling [43]

---

## Key Contributions

The paper makes three primary contributions:

1. **Novel Self-Supervised EEG Representation Learning Model:**
   - EEGMoE is a unified model capable of learning from unlabeled EEG data across various domains
   - A single pretrained model can be transferred to various downstream datasets
   - Enhances downstream task performance through transfer learning

2. **Domain-Decoupled MoE Architecture:**
   - First work to explore decoupled EEG representations using MoE
   - Disentangles domain-specific representations through specific experts with Top-K routing
   - Captures domain-shared representations through shared experts with soft routing

3. **State-of-the-Art Performance:**
   - Achieves superior performance across multiple benchmarks for ER, MI classification, and mental workload detection
   - Extensive experiments and visualizations demonstrate robust capability to learn both representation types

---

## Foundational Concepts

### 1. Electroencephalography (EEG)

**Definition:** EEG measures the electrical activity of the brain through electrodes placed on the scalp. The recorded signals reflect the synchronized activity of large populations of neurons, particularly pyramidal cells in the cerebral cortex [1].

**Key Characteristics:**
- **Frequency Bands:** EEG signals are typically analyzed in five canonical frequency bands:
  - **δ (Delta):** 1-4 Hz - associated with deep sleep
  - **θ (Theta):** 4-8 Hz - associated with drowsiness, meditation
  - **α (Alpha):** 8-14 Hz - associated with relaxed wakefulness
  - **β (Beta):** 14-31 Hz - associated with active thinking, focus
  - **γ (Gamma):** 31-45 Hz - associated with higher cognitive processing

- **Temporal Dynamics:** EEG signals are non-stationary, meaning their statistical properties change over time [27-30]

- **Spatial Distribution:** Different brain regions contribute differently to various tasks, creating distinct spatial patterns

### 2. Self-Supervised Learning

**Definition:** A learning paradigm where models learn representations from unlabeled data by solving pretext tasks (surrogate tasks) that don't require manual annotations [20, 44, 45].

**Common Pretext Tasks:**
- **Masked Signal Reconstruction:** Randomly mask portions of input and train model to reconstruct them (similar to BERT's masked language modeling)
- **Contrastive Learning:** Learn representations by distinguishing similar from dissimilar samples [71]
- **Temporal Prediction:** Predict future signal segments from past context
- **Vector-Quantized Prediction:** Predict discrete codes representing signal patches [22]

**Advantages for EEG:**
- Leverages vast amounts of unlabeled EEG data
- Learns generalizable representations transferable to downstream tasks
- Reduces dependency on expensive manual annotations

### 3. Transformer Architecture

**Definition:** Transformer is a deep learning architecture based on the self-attention mechanism, originally proposed for NLP [35] but successfully adapted to various domains including EEG [13, 67].

**Key Components:**

- **Multihead Self-Attention (MHSA):** Allows the model to attend to different positions in the input sequence simultaneously
  ```
  Attention(Q, K, V) = softmax(QK^T / √d_k)V
  ```
  where Q (query), K (key), V (value) are linear projections of the input

- **Feed-Forward Network (FFN):** Position-wise fully connected network applied after attention
  ```
  FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
  ```

- **Layer Normalization:** Stabilizes training by normalizing activations

- **Residual Connections:** Skip connections that help gradient flow in deep networks

**Why Transformers for EEG:**
- Capture long-range temporal dependencies
- Model global correlations in EEG signals
- Flexible architecture adaptable to various input formats

### 4. Mixture of Experts (MoE)

**Definition:** MoE is a neural network architecture that employs multiple "expert" subnetworks with a gating mechanism (router) that determines which experts to use for each input [50].

**Core Components:**

- **Experts:** Individual neural networks (typically MLPs) that process inputs
- **Router/Gating Network:** Computes activation probabilities for each expert
- **Routing Strategy:** Determines how experts are selected and combined

**Routing Strategies:**

1. **Top-K Routing:** Select the K experts with highest routing scores [53]
   ```
   p_i(x) = exp(g_i) / Σ_j exp(g_j)  where g = W_r · x
   ```
   Only top-K experts are activated per token

2. **Soft Routing:** Use all experts weighted by their activation probabilities
   - Provides smoother gradients
   - Learns from all data through all experts

**MoE in Large-Scale Models:**
- **Switch Transformers:** Scale to trillion parameters with sparse activation [54]
- **GShard:** Conditional computation with automatic sharding [55]
- **DeepSeek MoE:** Ultimate expert specialization in language models [36]

### 5. Domain Adaptation and Decoupling

**Domain:** A distinct distribution of data characterized by specific properties (e.g., different tasks, datasets, or subjects) [18, 24-26].

**Domain Adaptation:** Transfer knowledge from source domain(s) to a target domain [24].

**Domain Decoupling:** Learn separate representations for domain-specific and domain-shared features:
- **Domain-Shared Representations:** Common patterns across all domains
- **Domain-Specific Representations:** Unique characteristics of each domain

**Why Decoupling Matters:**
- Prevents negative transfer between dissimilar domains
- Preserves valuable domain-specific information
- Enables better generalization to new domains

### 6. Cross-Validation: Leave-One-Subject-Out (LOSO)

**Definition:** A validation strategy where data from one subject is held out as the test set while data from all other subjects is used for training [11, 12].

**Procedure:**
1. For N subjects, create N folds
2. In each fold, use one subject for testing, others for training
3. Average performance across all folds

**Advantages:**
- Evaluates model's ability to generalize to unseen subjects
- Critical for EEG applications where subject independence is essential
- Provides realistic estimate of real-world performance

### 7. Evaluation Metrics

**Accuracy (acc):** Proportion of correctly classified samples
```
acc = (TP + TN) / (TP + TN + FP + FN)
```

**Standard Deviation (std):** Measures variability across cross-validation folds

**AUROC (Area Under ROC Curve):** Measures classifier's ability to distinguish between classes across all thresholds
- Robust to class imbalance
- Values range from 0 to 1 (1 = perfect classifier)

**AUC-PR (Area Under Precision-Recall Curve):** Particularly useful for imbalanced datasets
- Focuses on performance on positive class
- More informative than AUROC when classes are imbalanced

---

## Methodology

### Overview of EEGMoE Architecture

The EEGMoE model consists of three main components (Figure 2):

1. **Spatial and Frequency Encoder:** Generates embeddings from preprocessed EEG data
2. **Domain-Decoupled Encoder:** Core component with SSMoE blocks for learning decoupled representations
3. **Downstream Classifier:** Simple linear layer for task-specific classification

### Data Preprocessing: 4-D Input Format

To incorporate temporal-spatial-frequency information and reduce differences from various EEG montages, raw EEG signals are converted to a unified 4-D format:

**Preprocessing Steps:**

1. **Segmentation:** Divide raw EEG signals into T-second non-overlapping segments

2. **Frequency Decomposition:** Decompose each segment into five frequency bands:
   - δ (1-4 Hz), θ (4-8 Hz), α (8-14 Hz), β (14-31 Hz), γ (31-45 Hz)

3. **2-D Brain Map Creation:** For each frequency band, map data into a 2-D grid representing electrode locations on the scalp (Figure 4)

4. **Smoothing:** Apply Gaussian filtering or z-score normalization to eliminate abrupt changes

5. **Stacking:** Stack 2-D brain maps along the frequency dimension

**Final Input Format:**
```
X ∈ R^((T×Sr)×B×H×W)
```
where:
- Sr = sampling rate
- B = number of frequency bands (5)
- H, W = height and width of 2-D brain maps
- T = time window length

This 4-D representation captures:
- **Temporal information** (T × Sr dimension)
- **Frequency information** (B dimension)
- **Spatial information** (H × W dimension)

### Domain-Decoupled Encoder

The core innovation of EEGMoE lies in its domain-decoupled encoder, which incorporates the **Specific and Shared MoE (SSMoE) block**.

#### Specific and Shared MoE (SSMoE) Block

The SSMoE block combines two types of expert groups:

```
SSMoE(x) = SpecMoE(x) + ShareMoE(x)
```

#### 1. Specific MoE (Domain-Specific Learning)

**Purpose:** Learn fine-grained, domain-specific representations through flexible expert combinations.

**Architecture:**
- Replaces the standard Transformer FFN with E expert networks {e_i}_(i=1)^|E|
- Each expert is a two-layer MLP with GELU activation

**Routing Mechanism:**

1. **Router computes scores:**
   ```
   g_x = W_e · x
   ```

2. **Activation probabilities via softmax:**
   ```
   p_i(x) = exp(g_xi) / Σ_(j=1)^|E| exp(g_xj)
   ```

3. **Top-K selection:** Choose K experts with highest probabilities

4. **Weighted output:**
   ```
   SpecMoE(x) = Σ_(i∈TopK) p_i(x) · e_i(x)
   ```

**Key Properties:**
- **Sparse activation:** Only K out of E experts are activated per token
- **Dynamic routing:** Different tokens can activate different expert combinations
- **Domain specialization:** Experts naturally specialize in different domains through training

#### 2. Shared MoE (Domain-Shared Learning)

**Purpose:** Learn domain-shared representations by processing all data through all experts.

**Architecture:**
- Uses F fixed expert networks {f_i}_(i=1)^|F|
- Same multihead self-attention module as Specific MoE

**Routing Mechanism:**

1. **Router computes scores** (same as Specific MoE):
   ```
   p_i(x) = exp(g_xi) / Σ_(j=1)^|F| exp(g_xj)
   ```

2. **All experts are used:** Unlike Top-K routing, all F experts process every token

3. **Weighted output:**
   ```
   ShareMoE(x) = Σ_(i∈F) p_i(x) · f_i(x)
   ```

**Key Properties:**
- **Dense activation:** All experts are activated for every token
- **Comprehensive learning:** Every expert learns from all domains
- **Shared knowledge:** Captures commonalities across domains

#### 3. Integration Strategy

The outputs of Specific and Shared MoE are combined through **additive fusion**:

```
SSMoE(x) = SpecMoE(x) + ShareMoE(x)
```

This simple addition effectively balances domain-specific and domain-shared representations.

### Training Framework

EEGMoE employs a two-stage training paradigm:

#### Stage 1: Self-Supervised Pretraining

**Objective:** Learn latent representations from large-scale unlabeled EEG data through masked signal reconstruction.

**Process (Figure 3):**

1. **Embedding Generation:**
   - Input data X passes through spatial and frequency encoder
   - Aggregates along channel and frequency dimensions
   - Produces embeddings Z

2. **Masking:**
   - Randomly mask a portion of embeddings (mask ratio = 0.4)
   - Creates corrupted input for SSMoE

3. **Reconstruction:**
   - SSMoE processes masked embeddings
   - Learns to reconstruct original embeddings Z̃

4. **Loss Function:**
   ```
   L_pretrain = L₁ + α·L_aux
   ```

   **a) Reconstruction Loss (L₁):**
   ```
   L₁ = (1/S) · Σ_(i=1)^S |Z_i - Z̃_i|
   ```
   - L₁ loss (Mean Absolute Error) for embedding reconstruction
   - S = number of tokens

   **b) Load-Balancing Auxiliary Loss (L_aux):**
   ```
   L_aux = E · Σ_(i=1)^E h_i · D_i
   ```
   
   Where:
   - h_i = fraction of tokens allocated to expert i
     ```
     h_i = (1/S) · Σ_(x∈B) 1{i ∈ TopK(d(x))}
     ```
   - D_i = fraction of router probability for expert i
     ```
     D_i = (1/S) · Σ_(x∈B) d_i(x)
     ```
   - α = 1 × 10⁻⁴ (hyperparameter controlling loss balance)

**Why Load Balancing is Critical:**

Without L_aux, Top-K routing can cause:
- **Expert collapse:** Most tokens allocated to few experts
- **Underutilization:** Some experts remain untrained
- **Training instability:** Imbalanced gradient updates

L_aux encourages uniform token distribution across experts.

#### Stage 2: Supervised Downstream Fine-Tuning

**Objective:** Transfer knowledge from pretraining to specific downstream tasks.

**Process:**

1. **Weight Inheritance:**
   - Load pretrained domain-decoupled encoder weights
   - Freeze or fine-tune encoder parameters

2. **Classifier Addition:**
   - Add simple linear classifier layer
   - No complex task-specific architecture needed

3. **Supervised Training:**
   - Train with labeled downstream dataset
   - Use cross-entropy loss:
   ```
   L_cls = -(1/N) · Σ_(i=1)^N Σ_(j=1)^C y_ij · log(ŷ_ij)
   ```
   where:
   - N = number of samples
   - C = number of categories
   - y_ij = true label (one-hot)
   - ŷ_ij = predicted probability

**Advantages:**
- **Simple fine-tuning:** Linear classifier is sufficient
- **Fast adaptation:** Minimal task-specific training needed
- **Knowledge transfer:** Leverages pretrained representations

---

## Experimental Setup

### Datasets

The study uses **nine datasets** across three EEG task categories (Table I):

#### Emotion Recognition (ER) Datasets

1. **DREAMER [56]:**
   - 14 channels, 128 Hz sampling rate
   - 23 participants, 18 film clips each
   - Duration: 65-393 seconds per clip
   - Based on dimensional emotion model

2. **SEED [57]:**
   - 62 channels, 200 Hz sampling rate
   - Based on discrete emotion model
   - Critical frequency bands and channels investigated

3. **MAHNOB-HCI [58]:**
   - 32 channels, 256 Hz sampling rate
   - 27 participants, 20 video clips
   - Duration: 35-117 seconds per clip
   - Multimodal database for affect recognition

4. **DEAP [32]:**
   - 32 channels, 128 Hz sampling rate
   - 32 participants, 40 music videos each
   - 3-second baseline + 60-second emotional signals
   - Widely used benchmark for emotion recognition

#### Motor Imagery (MI) Classification Datasets

5. **EEG Motor Movement/Imagery Dataset (EEGMIDB) [59]:**
   - 64 channels, 160 Hz sampling rate
   - 109 participants
   - 14 experimental runs per participant
   - Includes both motor movement and MI tasks

6. **BCI Competition IV-1 [60]:**
   - 59 channels, 100 Hz sampling rate
   - 7 participants
   - 2-class MI tasks, 100 trials each
   - 4-second trials

7. **BCI Competition IV-2a [33]:**
   - 22 channels, 250 Hz sampling rate
   - 9 participants
   - 4-class MI classification
   - 576 trials per participant (2 sessions)

#### Mental Workload (MWL) Detection Datasets

8. **EEGMat [61]:**
   - 23 channels, 500 Hz sampling rate
   - 36 participants
   - Mental arithmetic experiment (10 minutes)
   - Three phases: adaptation, resting, task

9. **STEW [34]:**
   - 14 channels, 128 Hz sampling rate
   - 48 participants
   - Multitasking workload experiment (5 minutes)
   - SIMKAP multitasking activity

### Dataset Splits

- **Pretraining (6 datasets):** DREAMER, SEED, MAHNOB-HCI, EEGMIDB, BCIC4-1, EEGMat
- **Fine-tuning & Validation (3 datasets):** DEAP, BCIC4-2a, STEW

### Implementation Details

**Hardware:**
- NVIDIA A800-SXM4-80G GPUs
- PyTorch library, Python 3.8.0

**Cross-Validation:**
- Leave-One-Subject-Out (LOSO) on DEAP, BCIC4-2a, STEW
- One subject as test set, remaining subjects as training set

**Model Hyperparameters (Tables II & III):**
- Specific experts: 4 (Top-K = 2)
- Shared experts: 2
- Mask ratio: 0.4
- α (load balancing weight): 1 × 10⁻⁴

### Baseline Models

The study compares EEGMoE against seven baseline models:

1. **EEGNet [66]:**
   - Compact CNN for EEG-based BCI
   - Uses depthwise and separable convolutions
   - Efficient feature extraction

2. **TSception [11]:**
   - Multiscale CNN for emotion recognition
   - Dynamic temporal layer for temporal-frequency representations
   - Asymmetric spatial layer for global/hemisphere representations

3. **EEG-Conformer [67]:**
   - Compact Convolutional Transformer
   - Convolution module for local features
   - Self-attention for global correlations

4. **LGGNet [12]:**
   - Neurologically inspired Graph Neural Network
   - Models relations within/between functional brain regions
   - Local-global-graph representations

5. **Seizure-MoE [41]:**
   - MoE framework for seizure subtype classification
   - Pretrained DNNs for feature extraction
   - MLPs as expert modules

6. **BIOT [46]:**
   - Self-supervised biosignal learning model
   - Tokenizes biosignals into "sentences"
   - Transformer architecture for cross-data learning

7. **LaBraM [22]:**
   - Large brain model for generic representations
   - Vector-quantized neural spectrum prediction
   - Neural tokenizer for EEG patches

---

## Results and Analysis

### Main Comparison Results

#### Emotion Recognition (DEAP Dataset) - Table IV

**Valence Recognition:**
- EEGMoE achieves superior accuracy compared to all baselines
- Outperforms state-of-the-art EEGFuseNet [62] by 2.96%
- Shows consistent improvement across all metrics (acc, AUC-PR, AUROC)

**Arousal Recognition:**
- EEGMoE outperforms baselines by 4.18%
- Demonstrates robust performance on both emotion dimensions

**Key Observation:** Models like EEGNet and EEG-Conformer show inconsistent performance across tasks, highlighting their task-specific limitations.

#### Motor Imagery (BCIC4-2a Dataset) - Table V

- EEGMoE achieves **6.01% accuracy improvement** over state-of-the-art DeepCNN [68]
- Consistently outperforms all baselines across all four MI classes
- Lower standard deviation indicates more stable performance

#### Mental Workload (STEW Dataset) - Table VI

- EEGMoE achieves **3.21% accuracy improvement** over STEW model [34]
- Superior performance on three-class workload classification
- Demonstrates generalization to cognitive state detection

### Key Findings

1. **Cross-Task Generalization:**
   - Single pretrained model transfers effectively to three distinct tasks
   - No task-specific architecture engineering required

2. **Comparison with Self-Supervised Models:**
   - Outperforms BIOT and LaBraM significantly
   - These models primarily trained on epilepsy data may not generalize well
   - Domain decoupling provides advantage over undecoupled pretraining

3. **Computational Efficiency:**
   - Sparse activation reduces inference cost
   - Comparable or lower computational load than equivalent dense models

---

## Ablation Studies

### 1. Effect of Large-Scale Pretraining - Table VII

**Comparison:**
- EEGMoE with pretrained weights vs. EEGMoE trained from scratch

**Results:**
- Pretrained model significantly outperforms from-scratch training across all tasks
- Demonstrates effectiveness of learning from large-scale unlabeled EEG data
- Domain-shared and domain-specific representations benefit downstream tasks

**Conclusion:** Pretraining is essential for capturing transferable representations.

### 2. Effect of Specific and Shared Experts - Table VIII

**Configurations Tested:**
- **Full Model:** Top-K=2, Shared=2 (default)
- **No Specific:** Top-K=0, Shared=4 (only shared experts)
- **No Shared:** Top-K=4, Shared=0 (only specific experts)

**Results:**
- Removing either component degrades performance on all datasets
- **No Specific:** Impairs ability to select domain-appropriate expert combinations
- **No Shared:** Prevents comprehensive learning of domain-shared representations

**Conclusion:** Both specific and shared experts are indispensable for optimal performance.

### 3. Effect of SSMoE Architecture - Table IX

**Comparison Models:**
- **EEGMoE:** Full SSMoE with domain decoupling
- **Activated Equivalent:** Same number of activated parameters during inference
- **Total Equivalent:** Same total parameters as EEGMoE training

**Results:**
- EEGMoE outperforms both equivalent models despite comparable computational load
- Non-SSMoE models lack domain decoupling capability

**Conclusion:** Domain decoupling (not just increased capacity) drives performance gains.

### 4. Parameter Analysis

#### Number of Experts - Table X

**Configurations Tested:**
- Various combinations of specific experts and Top-K values
- Different numbers of shared experts

**Findings:**
- **Top-K routing:** Small values (e.g., 2) are sufficient
  - Allows combination of multiple experts without redundancy
- **Shared experts:** More than one improves generalization
  - Ensures commonalities are adequately represented
- **Moderation is key:** Too few or too many experts reduces effectiveness

**Practical Recommendations:**
- Top-K = 2 for specific experts
- At least 2 shared experts
- Total experts should match task complexity

#### Pretraining Objectives - Table XI

**Objectives Tested:**
- **L₁ loss:** Mean absolute error reconstruction (default)
- **L₂ loss:** Mean squared error reconstruction
- **Cosine loss:** Cosine similarity reconstruction
- **Contrastive loss (RP):** Relative positioning-based contrastive learning [71]
- **Temporal prediction (TS):** Temporal shuffling prediction

**Results:**
- Performance differences across objectives are relatively small
- L₁ loss selected as default (simplest and most effective)

**Conclusion:** Architecture is flexible and robust to different pretraining paradigms.

#### Expert Fusion Strategies - Table XII

**Strategies Compared:**
- **Additive Fusion:** Simple summation (default)
- **Gated Fusion:** Gating network computes relative weights [72]
- **Attention-based Fusion:** Transformer layer for fusion

**Results:**
- Additive fusion consistently yields best performance
- Simple summation is sufficient for balancing representations

**Conclusion:** Complex fusion mechanisms are unnecessary; additive fusion works best.

#### Mask Ratio - Table XIII

**Ratios Tested:** 0.2, 0.4, 0.6, 0.8

**Results:**
- **Optimal:** Mask ratio = 0.4
- Performance first improves then declines as ratio increases
- **Too low (0.2):** Insufficient training, inadequate representation learning
- **Too high (0.8):** Reconstruction task too challenging, impedes training

**Conclusion:** Moderate masking (40%) provides optimal training signal.

---

## Visualization and Interpretation

### Expert Activation Analysis

The paper provides two complementary visualizations to demonstrate domain decoupling:

#### Figure 5: Task Preferences for Experts

**What it shows:** Activation proportions of each expert within the specific expert group across three tasks.

**Key Observations:**

- **Emotion Recognition:**
  - Higher activation for Experts 1, 3, and 5
  - These experts specialize in emotional processing patterns

- **Motor Imagery Classification:**
  - Higher activation for Experts 1, 4, and 6
  - Notably lower activation for Experts 3 and 5
  - Different expert combination than ER

- **Mental Workload Detection:**
  - Higher activation for Expert 2
  - Lower activation for Expert 1
  - Distinctly different pattern from both ER and MI

**Interpretation:** Different tasks display distinct preferences for different experts, confirming that the model learns task-specific processing pathways.

#### Figure 6: Expert Responsibilities

**What it shows:** Percentage of data from each task handled by each expert.

**Key Observations:**

- **Experts 1, 3, 5:** Primarily responsible for Emotion Recognition
- **Experts 2, 4:** Mainly handle Mental Workload Detection
- **Expert 6:** Primarily used for Motor Imagery Classification

**Interpretation:** Individual experts specialize in distinct tasks, demonstrating successful domain decoupling through expert specialization.

### Joint Interpretation

**Figure 5 answers:** "Which experts does each task prefer?"
**Figure 6 answers:** "Which task is each expert mainly responsible for?"

**Together, they demonstrate:**

1. **Bidirectional Specialization:**
   - Tasks consistently activate specific expert subsets
   - Experts consistently specialize in specific tasks

2. **Effective Decoupling:**
   - Different tasks engage different neural pathways
   - No single expert is overloaded with all tasks

3. **Necessity of Decoupling:**
   - Distinct activation patterns reveal intrinsic task differences
   - Undecoupled models would miss these specialized representations

---

## Critical Analysis and Future Directions

### Strengths

1. **Novel Architecture:**
   - First application of domain-decoupled MoE to EEG representation learning
   - Elegant solution to gradient conflict problem in multi-domain training

2. **Comprehensive Evaluation:**
   - Nine datasets across three task categories
   - Extensive ablation studies validating each design choice
   - Multiple evaluation metrics for robust assessment

3. **Practical Impact:**
   - Single pretrained model transfers to multiple downstream tasks
   - Reduces need for task-specific model engineering
   - Leverages unlabeled EEG data effectively

4. **Interpretability:**
   - Expert activation visualizations provide insight into model behavior
   - Clear demonstration of domain decoupling mechanism

### Limitations

1. **Computational Requirements:**
   - Large-scale pretraining requires significant GPU resources
   - May be challenging for smaller research groups to replicate

2. **Task Coverage:**
   - Limited to three mainstream EEG tasks (ER, MI, MWL)
   - Other important applications (seizure detection, sleep staging) not evaluated

3. **Subject Diversity:**
   - While using LOSO cross-validation, demographic diversity of subjects not extensively analyzed
   - Cross-cultural generalization remains unexplored

4. **Comparison Fairness:**
   - Some baselines may not be fully optimized
   - Reproduced baselines (†marked) may differ from original implementations

### Future Directions

The authors outline several promising research directions:

1. **Expanded Task Coverage:**
   - **Neuromodulation:** Brain stimulation applications
   - **Brain-Controlled Robotics:** Motor prosthetics and assistive devices
   - **Fatigue Detection:** Monitoring alertness in safety-critical domains
   - **Clinical Diagnostics:** Neurological disorder detection and monitoring

2. **Architecture Improvements:**
   - More sophisticated routing mechanisms
   - Hierarchical expert organization
   - Dynamic expert creation/pruning

3. **Multimodal Integration:**
   - Combine EEG with other physiological signals (ECG, EMG, EOG)
   - Leverage complementary information from multiple modalities

4. **Foundation Model Development:**
   - Scale to larger model sizes and datasets
   - Explore emergent capabilities in EEG foundation models
   - Investigate in-context learning for EEG tasks

---

## References and Citations

### EEG Fundamentals

[1] A. Biasiucci, B. Franceschiello, and M. M. Murray, "Electroencephalography," *Current Biology*, vol. 29, no. 3, pp. R80-R85, 2019.

[2] X. Wang, D. Wang, X. Gao, Y. Zhao, and S. C. Chiu, "Enhancing EEG-based decision-making performance prediction by maximizing mutual information between emotion and decision-relevant features," *IEEE Transactions on Affective Computing*, vol. 15, no. 3, pp. 1228-1240, Jul. 2024.

### EEG Applications

[3] Z. Zhang, S. Zhong, and Y. Liu, "Beyond mimicking under-represented emotions: Deep data augmentation with emotional subspace constraints for EEG-based emotion recognition," *AAAI Conference on Artificial Intelligence*, 2024.

[4] S. An, S. Kim, P. Chikontwe, and S. H. Park, "Dual attention relation network with fine-tuning for few-shot EEG motor imagery classification," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 11, pp. 15479-15493, Nov. 2024.

[5] Z. Wang, Y. Ouyang, and H. Zeng, "ARFN: An attention-based recurrent fuzzy network for EEG mental workload assessment," *IEEE Transactions on Instrumentation and Measurement*, vol. 73, pp. 1-14, 2024.

### Traditional EEG Deep Learning

[6] F. Shen, G. Dai, G. Lin, J. Zhang, W. Kong, and H. Zeng, "EEG-based emotion recognition using 4D convolutional recurrent neural network," *Cognitive Neurodynamics*, vol. 14, no. 6, pp. 815-828, 2020.

[11] Y. Ding, N. Robinson, S. Zhang, Q. Zeng, and C. Guan, "TSception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition," *IEEE Transactions on Affective Computing*, vol. 14, no. 3, pp. 2238-2250, Jul. 2023.

[12] Y. Ding, N. Robinson, C. Tong, Q. Zeng, and C. Guan, "LGGNet: Learning from local-global-graph representations for brain-computer interface," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 7, pp. 9773-9786, Jul. 2024.

[66] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, Oct. 2018.

[67] Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional Transformer for EEG decoding and visualization," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 710-719, 2023.

### Self-Supervised EEG Learning

[20] M. H. Rafiei, L. V. Gauthier, H. Adeli, and D. Takabi, "Self-supervised learning for electroencephalography," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 2, pp. 1457-1471, Feb. 2024.

[21] K. Yi, Y. Wang, K. Ren, and D. Li, "Learning topology-agnostic EEG representations with geometry-aware modeling," *Neural Information Processing Systems*, vol. 36, 2023.

[22] W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large brain model for learning generic representations with tremendous EEG data in BCI," arXiv:2405.18765, 2024.

[23] Y. Zhang et al., "DMAE-EEG: A pretraining framework for EEG spatiotemporal representation learning," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 36, no. 10, pp. 17664-17678, Oct. 2025.

[46] C. Yang, M. Westover, and J. Sun, "BIOT: Biosignal transformer for cross-data learning in the wild," *Neural Information Processing Systems*, vol. 36, 2023.

[71] H. Banville, O. Chehab, A. Hyvärinen, D.-A. Engemann, and A. Gramfort, "Uncovering the structure of clinical EEG signals with self-supervised learning," *Journal of Neural Engineering*, vol. 18, no. 4, Aug. 2021.

### Mixture of Experts

[36] D. Dai et al., "DeepSeek MoE: Towards ultimate expert specialization in mixture-of-experts language models," arXiv:2401.06066, 2024.

[50] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Computation*, vol. 3, no. 1, pp. 79-87, Mar. 1991.

[53] N. Shazeer et al., "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," arXiv:1701.06538, 2017.

[54] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1-39, 2022.

[55] D. Lepikhin et al., "GShard: Scaling giant models with conditional computation and automatic sharding," arXiv:2006.16668, 2020.

### Transformer Architecture

[35] A. Vaswani et al., "Attention is all you need," *Neural Information Processing Systems*, vol. 30, 2017.

### Domain Adaptation

[18] Y. Wang, B. Zhang, and Y. Tang, "DMMR: Cross-subject domain generalization for EEG-based emotion recognition via denoising mixed mutual reconstruction," *AAAI Conference on Artificial Intelligence*, 2024.

[24] Y. Zhou et al., "Cross-subject cognitive workload recognition based on EEG and deep domain adaptation," *IEEE Transactions on Instrumentation and Measurement*, vol. 72, pp. 1-12, 2023.

### Datasets

[32] S. Koelstra et al., "DEAP: A database for emotion analysis using physiological signals," *IEEE Transactions on Affective Computing*, vol. 3, no. 1, pp. 18-31, Jan. 2012.

[33] C. Brunner, R. Leeb, G. Müller-Putz, A. Schlögl, and G. Pfurtscheller, "BCI competition 2008 - Graz dataset a," *Institute for Knowledge Discovery (Lab. Brain-Computer Interfaces), Graz University of Technology*, vol. 16, pp. 1-6, Jan. 2008.

[34] W. L. Lim, O. Sourina, and L. P. Wang, "STEW: Simultaneous task EEG workload data set," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 26, no. 11, pp. 2106-2114, Nov. 2018.

[56] S. Katsigiannis and N. Ramzan, "DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices," *IEEE Journal of Biomedical and Health Informatics*, vol. 22, no. 1, pp. 98-107, Jan. 2018.

[57] W.-L. Zheng and B.-L. Lu, "Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks," *IEEE Transactions on Autonomous Mental Development*, vol. 7, no. 3, pp. 162-175, Sep. 2015.

[58] M. Soleymani, J. Lichtenauer, T. Pun, and M. Pantic, "A multimodal database for affect recognition and implicit tagging," *IEEE Transactions on Affective Computing*, vol. 3, no. 1, pp. 42-55, Jan. 2012.

[59] G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw, "BCI2000: A general-purpose brain-computer interface (BCI) system," *IEEE Transactions on Biomedical Engineering*, vol. 51, no. 6, pp. 1034-1043, Jun. 2004.

[60] B. Blankertz, G. Dornhege, M. Krauledat, K.-R. Müller, and G. Curio, "The non-invasive Berlin brain-computer interface: Fast acquisition of effective performance in untrained subjects," *NeuroImage*, vol. 37, no. 2, pp. 539-550, Aug. 2007.

[61] I. Zyma et al., "Electroencephalograms during mental arithmetic task performance," *Data*, vol. 4, no. 1, p. 14, Jan. 2019.

### Related EEG MoE Works

[40] L. Yang et al., "EEG emotion recognition via identity based multi-gate mixture-of-experts network," *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Dec. 2022.

[41] Z. Du, R. Peng, W. Liu, W. Li, and D. Wu, "Mixture of experts for EEG-based seizure subtype classification," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 4781-4789, 2023.

[42] X.-H. Liu, W.-B. Jiang, W.-L. Zheng, and B.-L. Lu, "MoGE: Mixture of graph experts for cross-subject emotion recognition via decomposing EEG," *IEEE International Conference on Bioinformatics and Biomedicine (BIBM)*, Dec. 2024.

[43] Y. Gui, M. Chen, Y. Su, G. Luo, and Y. Yang, "EEGMamba: Bidirectional state space model with mixture of experts for EEG multi-task classification," arXiv:2407.20254, 2024.

### Additional References

[31] B. Liu, X. Liu, X. Jin, P. Stone, and Q. Liu, "Conflict-averse gradient descent for multi-task learning," *Neural Information Processing Systems*, vol. 34, 2021.

[72] X. Zhang et al., "A foundation model for lesion segmentation on brain MRI with mixture of modality experts," *IEEE Transactions on Medical Imaging*, vol. 44, no. 6, pp. 2594-2604, Jun. 2025.

---

## Appendix: Key Formulas Summary

### SSMoE Block

```
SSMoE(x) = SpecMoE(x) + ShareMoE(x)
```

### Specific MoE (Top-K Routing)

```
g_x = W_e · x
p_i(x) = exp(g_xi) / Σ_j exp(g_xj)
SpecMoE(x) = Σ_(i∈TopK) p_i(x) · e_i(x)
```

### Shared MoE (Soft Routing)

```
ShareMoE(x) = Σ_(i∈F) p_i(x) · f_i(x)
```

### Pretraining Loss

```
L_pretrain = L₁ + α·L_aux
L₁ = (1/S) · Σ_i |Z_i - Z̃_i|
L_aux = E · Σ_i h_i · D_i
```

### Fine-tuning Loss

```
L_cls = -(1/N) · Σ_i Σ_j y_ij · log(ŷ_ij)
```

---

**Document Version:** 1.0  
**Last Updated:** March 27, 2026  
**Prepared for:** Replicate Research Project
