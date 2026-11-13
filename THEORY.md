# Technical Theory & Algorithmic Architecture

This document provides a rigorous mathematical and algorithmic explanation of the core pipelines in **Vision-Based Chess AI**.

---

## Table of Contents
1. [Computer Vision & Perspective Correction](#1-computer-vision--perspective-correction)
   - [Planar Homography & Matrix Transformation](#planar-homography--matrix-transformation)
   - [Checkerboard Grid Energy Localization](#checkerboard-grid-energy-localization)
   - [Sub-Square Decomposition](#sub-square-decomposition)
2. [Deep Learning Spatial Classification](#2-deep-learning-spatial-classification)
   - [Model Architecture (MobileNetV3-Small)](#model-architecture-mobilenetv3-small)
   - [Loss Formulation & Label Smoothing](#loss-formulation--label-smoothing)
   - [Domain Randomization & Synthetic Augmentation](#domain-randomization--synthetic-augmentation)
3. [State Inference & Move Disambiguation](#3-state-inference--move-disambiguation)
   - [Graph Transition Formulation](#graph-transition-formulation)
   - [Temporal Filtering & Stability Windows](#temporal-filtering--stability-windows)
4. [Engine Evaluation Architectures](#4-engine-evaluation-architectures)
   - [UCI Stockfish Search & Alpha-Beta Pruning](#uci-stockfish-search--alpha-beta-pruning)
   - [Dual Policy/Value Convolutional Network](#dual-policyvalue-convolutional-network)
   - [Monte Carlo Tree Search (MCTS) Formulation](#monte-carlo-tree-search-mcts-formulation)

---

## 1. Computer Vision & Perspective Correction

### Planar Homography & Matrix Transformation
A physical or rendered chess board represents a planar surface in projective space $\mathbb{P}^2$. Given any viewing angle (e.g. tilted webcam, angled monitor capture), the relationship between points on the image plane $\mathbf{x} = [u, v, 1]^T$ and the canonical top-down board plane $\mathbf{X} = [x, y, 1]^T$ is governed by a 2D projective transformation (homography matrix $\mathbf{H} \in \mathbb{R}^{3 \times 3}$):

$$\mathbf{x} \sim \mathbf{H} \mathbf{X} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Because $\mathbf{H}$ has 8 degrees of freedom up to scale, it can be uniquely recovered from 4 corresponding point pairs $(u_i, v_i) \leftrightarrow (x_i, y_i)$ using the **Direct Linear Transformation (DLT)** algorithm by solving the homogeneous linear system:

$$\mathbf{A} \mathbf{h} = \mathbf{0}$$

Once $\mathbf{H}$ is estimated, the inverse transformation $\mathbf{H}^{-1}$ is applied via bilinear interpolation to warp the arbitrary quadrilateral into an orthographic $512 \times 512$ pixel image.

### Checkerboard Grid Energy Localization
For screen capture environments where the board is axis-aligned, an edge-energy metric is evaluated across candidate bounding boxes. A valid chess board exhibits strong internal gradient periodicities at step intervals of $w/8$ and $h/8$. 

Using vertical and horizontal Sobel kernels $S_x, S_y$:

$$G_x = I * S_x, \quad G_y = I * S_y, \quad |\nabla I| = |G_x| + |G_y|$$

The internal grid energy score $E(B)$ for a bounding box $B = (x, y, w, h)$ is defined by integrating edge magnitude along the 7 internal column and row dividers:

$$E(B) = \frac{1}{7} \sum_{k=1}^{7} \left( \frac{1}{h} \int_{y}^{y+h} |\nabla I|(x + k\frac{w}{8}, \hat{y}) d\hat{y} + \frac{1}{w} \int_{x}^{x+w} |\nabla I|(\hat{x}, y + k\frac{h}{8}) d\hat{x} \right)$$

The bounding box maximizing $E(B)$ subject to aspect ratio constraint $0.95 \le w/h \le 1.05$ is locked as the canonical board region.

### Sub-Square Decomposition
The normalized $512 \times 512$ image is divided into an $8 \times 8$ grid of cells $C_{r,c}$ where each square is $64 \times 64$ pixels:

$$C_{r,c} = I_{\text{warped}}[r \cdot 64 : (r+1) \cdot 64, \; c \cdot 64 : (c+1) \cdot 64]$$

To eliminate boundary border artifacts caused by thick square lines or neighboring piece silhouettes, an internal inset crop $\delta = 3\text{px}$ is applied prior to classification.

---

## 2. Deep Learning Spatial Classification

### Model Architecture (MobileNetV3-Small)
The 13-class square classifier utilizes **MobileNetV3-Small** pre-trained on ImageNet and fine-tuned for high-speed, low-latency CPU and GPU inference.

* **Inverted Residual Blocks (MBConv):** Uses depthwise separable convolutions with squeeze-and-excitation (SE) attention to minimize parameter count while maintaining expressive spatial feature maps.
* **Hard-Swish Activation:**
  $$\text{h-swish}(x) = x \frac{\text{ReLU6}(x + 3)}{6}$$
* **Classification Head:** Linear projection from the penultimate pooled feature vector $\mathbf{z} \in \mathbb{R}^{576}$ to logits $\mathbf{z}_{\text{out}} \in \mathbb{R}^{13}$:
  $$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_c \mathbf{z} + \mathbf{b}_c)$$

Classes include:
`0: empty`, `1..6: White {P, N, B, R, Q, K}`, `7..12: Black {p, n, b, r, q, k}`.

### Loss Formulation & Label Smoothing
Training optimizes the Cross-Entropy loss with label smoothing parameter $\alpha = 0.05$:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{13} q_k \log(p_k), \quad \text{where } q_k = (1 - \alpha) \cdot \mathbb{I}(y = k) + \frac{\alpha}{13}$$

Label smoothing prevents the network from becoming overconfident on synthetic borders, encouraging generalizable features across subtle lighting variations and highlight washes.

### Domain Randomization & Synthetic Augmentation
To bridge the simulation-to-reality gap between rendered sprites and screen captures:
* **Photometric jitter:** Brightness ($[0.85, 1.15]$), contrast ($[0.85, 1.15]$), and Gaussian sensor noise ($\sigma \sim \mathcal{N}(0, 4)$).
* **Highlight Emulation:** Random additive alpha washes simulating Chess.com last-move yellow highlights ($[247, 236, 116]$), check indicator red glow ($[232, 65, 24]$), and pre-move cyan overlays.

---

## 3. State Inference & Move Disambiguation

### Graph Transition Formulation
Given a previous valid board state $S_{t-1}$ with piece placement $P(S_{t-1})$, and a newly classified visual piece matrix $\hat{P}_t$, the legal transition move $m^* \in \text{LegalMoves}(S_{t-1})$ is formulated as:

$$m^* = \arg\min_{m \in \text{LegalMoves}(S_{t-1})} \mathcal{D}(P(S_{t-1} \circ m), \hat{P}_t)$$

where $\mathcal{D}(P_A, P_B) = \sum_{r=0}^{7} \sum_{c=0}^{7} \mathbb{I}(P_A[r,c] \neq P_B[r,c])$ is the Hamming distance between placements.
* If $\min \mathcal{D} = 0$, an exact legal transition is confirmed.
* If $\min \mathcal{D} \le 2$ (allowing for 1-square classification noise), the move is accepted if confidence exceeds the transition threshold.

### Temporal Filtering & Stability Windows
To prevent triggering moves during in-flight screen animations:
$$\text{StableFrames}(t) = \begin{cases} \text{StableFrames}(t-1) + 1 & \text{if } \hat{P}_t = \hat{P}_{t-1} \\ 0 & \text{otherwise} \end{cases}$$
Moves are only inferred and dispatched when $\text{StableFrames}(t) \ge N_{\text{threshold}}$ (default: $N = 3$), ensuring all sliding animations, clock toggles, and visual updates have concluded.

---

## 4. Engine Evaluation Architectures

### UCI Stockfish Search & Alpha-Beta Pruning
Stockfish implements an advanced **Negamax Alpha-Beta search** enhanced with:
* **Principal Variation Search (PVS)**
* **Null Move Pruning & Late Move Reductions (LMR)**
* **NNUE (Efficiently Updatable Neural Network):** Evaluates leaf positions via a 4-layer quantized network trained on billions of game states:
  $$\text{Score} = \mathbf{W}_3 \cdot \text{ClippedReLU}(\mathbf{W}_2 \cdot \text{ClippedReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + b_3$$

### Dual Policy/Value Convolutional Network
The custom engine implements an AlphaZero-style architecture:
* **Input Tensor $\mathbf{X} \in \mathbb{R}^{8 \times 8 \times 18}$:** 12 piece planes + side to move + castling rights (4 planes) + en passant target.
* **Residual Backbone:** 6 to 12 residual blocks with batch normalization and ReLU.
* **Policy Head $\mathbf{p}$:** Outputs probability distribution over all $8 \times 8 \times 73 = 4672$ possible move representations:
  $$\mathbf{p} = \text{softmax}(\mathbf{W}_p \mathbf{z}_{\text{conv}} + \mathbf{b}_p)$$
* **Value Head $v$:** Scalar evaluation $v \in [-1, 1]$ predicting win probability:
  $$v = \tanh(\mathbf{W}_v \mathbf{z}_{\text{dense}} + b_v)$$

### Monte Carlo Tree Search (MCTS) Formulation
During MCTS simulations, node selection follows the **PUCT (Polynomial Upper Confidence Trees)** formula:

$$a_t = \arg\max_{a} \left( Q(s, a) + U(s, a) \right)$$

$$U(s, a) = c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

where:
* $Q(s, a)$ is the mean action value from previous rollouts.
* $P(s, a)$ is the prior probability predicted by the policy head.
* $N(s, a)$ is the visit count for the action.
* $c_{\text{puct}}$ balances exploitation vs. exploration.
