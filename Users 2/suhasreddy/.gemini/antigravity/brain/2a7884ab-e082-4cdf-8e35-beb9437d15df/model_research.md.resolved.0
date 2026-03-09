# Smart City — Model Architecture Deep Research

> **Purpose:** Before building anything, this document answers: *What model choices would make this project genuinely novel vs. just assembling off-the-shelf tools?*

---

## The Core Problem with Stock Models

| Component | Stock Choice | Why It's Not Impressive |
|---|---|---|
| Detection | YOLOv11 fine-tuned on UA-DETRAC | Every CV engineer has done YOLO fine-tuning. It's expected, not novel. |
| Tracking | BoT-SORT via BoxMOT library | You're calling `pip install boxmot` and passing a flag. No contribution. |
| Prediction | LSTM on vehicle counts | LSTMs are 2015 technology. Dated for 2026. |
| Anomaly | VAE on detection counts | Basic, well-trodden approach. |

**A senior engineer will see through this in 30 seconds.** Below is what cutting-edge research looks like in each domain, and where you can make a **genuine contribution** that's feasible on Colab.

---

## 1. Detection — Beyond Vanilla YOLO

### What's Cutting-Edge (2024-2025)

| Model/Paper | What's Novel | Published |
|---|---|---|
| **ECL-YOLOv11** | Adds 3 modules to YOLOv11: (1) Sobel edge-enhancement for boundary retention in fog/rain, (2) Context-guided multi-scale fusion (AENet) for small objects, (3) Lightweight shared detection head. 237.8 FPS. | MDPI 2024 |
| **YOLOv11-TWCS** | Integrates TransWeather image restoration + CBAM attention + spatial-channel downsampling. Restores degraded images before detection. | 2024 |
| **DA-RAW** | Unsupervised domain adaptation: separates style gap vs weather gap. Style alignment (adversarial) + weather alignment (contrastive learning). No labels needed for target domain. | ECCV 2024 |
| **RT-DETR** | Transformer-based real-time detector. No NMS needed (end-to-end). Competitive with YOLO but architecturally different. | CVPR 2024 |
| **Weather-Adaptive Attention** | Dynamic feature reweighting based on visual degradation cues. Hybrid CNN-ViT architecture. | 2024-2025 |

### Your Novel Contribution: **Weather-Conditioned Adaptive Detection**

Instead of just fine-tuning YOLOv11, build a **weather-aware detection pipeline** with a novel component:

```
Frame → Weather Classifier → Condition-Specific Detection Head
         (ResNet-18)           (separate fine-tuned weights per condition)
```

**What makes this novel:**
1. **Train separate detection heads** (or final layers) for different weather conditions — not just adjusting confidence thresholds, but actually switching model weights
2. **Inspired by ECL-YOLOv11** — add a Sobel edge-enhancement module to the YOLO backbone specifically for rain/fog frames
3. **Benchmark properly** — compare your weather-adaptive approach vs vanilla YOLOv11 vs ECL-YOLOv11 across clear/rain/fog/night conditions on Singapore data

**Why this is feasible:**
- The weather classifier is tiny (ResNet-18, trains in <1 hour on Colab)
- Edge enhancement is a simple conv layer addition, not a full architecture redesign
- You're not inventing a new detector — you're creating a **novel routing/adaptation mechanism** on top of a proven detector
- Publishable result: *"Weather-adaptive detection routing improves mAP by X% in rain and Y% at night on Singapore traffic cameras"*

**Feasibility on Colab T4:** ✅ All components fit in 15GB VRAM. Training per condition head: ~2-3 hours each.

---

## 2. Tracking — Beyond Plug-and-Play BoxMOT

### What's Cutting-Edge (2024-2025)

| Model/Paper | What's Novel | Published |
|---|---|---|
| **MOTIP** | Reframes MOT as in-context ID prediction. Transformer decoder directly predicts track IDs from historical trajectory context. | CVPR 2025 |
| **MASA** | Matching Anything by Segmenting Anything. Uses SAM for appearance matching, works on any domain. | CVPR 2024 |
| **Hybrid-SORT** | Combines motion + appearance with a weak-cues association strategy. AAAI 2024. | AAAI 2024 |
| **GNN-based MOT** | Graph neural networks for joint detection + data association. Nodes = detections, edges = associations. Message passing captures higher-order relationships. | Various 2024 |

### Your Novel Contribution: **Graph-Based Camera-Network Tracking**

BoxMOT handles single-camera tracking well. Your novelty is **cross-camera association** on a real camera network:

1. **Single-camera:** Use BoT-SORT (off-the-shelf, this is fine — the novelty is elsewhere)
2. **Cross-camera:** Build a **spatial-temporal graph** where:
   - **Nodes** = tracked vehicles exiting one camera's field of view
   - **Edges** = potential re-identification matches at adjacent cameras
   - **Edge weights** = cosine similarity of Re-ID embeddings × spatial plausibility (travel time between cameras based on GPS distance)
3. **Novel GNN layer:** Use a simple Graph Attention Network (GAT) to learn which cross-camera matches are correct

**What makes this novel:**
- Cross-camera vehicle Re-ID on a real city-scale camera network is a **genuinely hard research problem**
- You're combining appearance similarity (Re-ID embeddings) with spatial-temporal constraints (GPS location + expected travel time)
- No one has published cross-camera Re-ID results on Singapore's LTA camera network

**Feasibility on Colab T4:** ✅ The GNN is tiny (few thousand parameters). The expensive part is Re-ID embedding extraction, which runs at inference speed. You can start with pre-trained OSNet and only train the GNN matching layer.

> [!WARNING]
> Cross-camera Re-ID is genuinely hard and may not work well. **That's okay.** Documenting why it fails (camera resolution mismatch, large GPS gaps between cameras, appearance changes from different angles) is equally impressive as documenting success. This shows research maturity.

---

## 3. Prediction — Beyond Basic LSTM

### What's Cutting-Edge (2024-2025)

| Model/Paper | What's Novel | Published |
|---|---|---|
| **STGformer** | STG-attention mechanism captures global+local spatio-temporal interactions in one layer. Merges GCN + Transformer strengths. | Oct 2024 |
| **GE-STT** | Graph-Enhanced Spatial-Temporal Transformer. GCN+GRU module feeds into Transformer. Uses original data as correction term. | Feb 2025 |
| **xMTrans** | Cross-modality Transformer for traffic. Multi-head attention fuses weather + traffic + external data. | 2024 |
| **STTGCN** | Replaces Transformer's MLP with Kolmogorov-Arnold Network (KAN) for interpretability + BiLSTM for temporal. | Nov 2025 |
| **CAFMGCN** | Cross-Attention Fusion Multi-Graph Convolutional Network. Fuses spatial and temporal with cross-attention. | 2024 |
| **MC-STTM** | Multi-Channel Spatial-Temporal Transformer. GCN spatial features + Transformer temporal + adaptive adjacency matrix. | May 2024 |

### Your Novel Contribution: **Weather-Traffic Cross-Modal Graph Transformer**

This is where your **multi-modal data** becomes the actual model innovation:

1. **Build a spatial graph:** Nodes = 87 cameras. Edges = road connections (cameras on same expressway are connected). Edge weight = GPS distance.
2. **Node features at each timestep:** Vehicle count, mean detection confidence, vehicle class distribution, weather condition (one-hot), temperature, PM2.5, taxi density nearby
3. **Architecture:**
   ```
   Multi-Modal Node Encoder (FC layers per modality)
        ↓
   Graph Attention Network (spatial message passing)
        ↓
   Temporal Transformer (multi-head self-attention over time steps)
        ↓  
   Cross-Attention Fusion (weather sequence ⊗ traffic sequence)
        ↓
   Prediction Head (vehicle count + congestion level, 15/30 min ahead)
   ```
4. **Novel component: Cross-Attention Fusion Block** — learns to attend to weather features when predicting traffic, capturing correlations like *"rain forecast → congestion increase on PIE 20 min later"*

**What makes this novel:**
- No existing traffic prediction model fuses **real-time weather + taxi demand + air quality + camera detection confidence** as node features in a graph
- The cross-attention fusion between weather and traffic sequences is inspired by xMTrans but applied to a **real government camera network with verified multi-modal APIs**
- You're not just predicting traffic — you're modeling **weather-traffic causation** (publishable insight)

**Feasibility on Colab T4:** ✅ Graph has only 87 nodes. Temporal window of ~180 timesteps (3 hours at 1-min intervals). This is a tiny model. Trains in hours, not days.

---

## 4. Anomaly Detection — Beyond Basic VAE

### What's Cutting-Edge (2024-2025)

| Model/Paper | What's Novel | Published |
|---|---|---|
| **uTRAND** | Redefines anomaly detection in semantic-topological space. Patch-based graphs at intersections. Explainable rules. | Apr 2024 |
| **GeoGNFTOD** | GNN + VAE for online trajectory anomaly. Maps trajectories onto road network graphs. Transformer encoding. | Aug 2024 |
| **MTGAE** | Mirror Temporal Graph Autoencoder. Captures spatio-temporal correlations between road network nodes. | Jan 2024 |
| **GETAD** | Graph-enhanced trajectory anomaly. Integrates road topology + segment semantics + historical patterns. | Sep 2025 |
| **TSGAD** | Fuses prediction-based + reconstruction-based approaches using VAE for trajectory anomaly. | Apr 2024 |

### Your Novel Contribution: **Multi-Scale Anomaly Detection with Graph Context**

Instead of just detecting "unusual count" anomalies, build a **spatially aware anomaly detector** that understands the camera network:

1. **Frame-level anomaly:** Per-camera anomaly score based on detection confidence distribution (autoencoder reconstruction error)
2. **Trajectory-level anomaly:** Vehicle paths that deviate from learned normal flow patterns (wrong-way driving, erratic movement)
3. **Network-level anomaly (NOVEL):** Cross-camera anomaly detection — a traffic jam on camera A should be "expected" 10 min later on downstream camera B. If B shows normal traffic when A is jammed → possible road closure, accident, or route change.

**Architecture:**
```
Per-Camera Autoencoder (reconstruction-based)
       ↓
Temporal Anomaly Score Sequence
       ↓
Graph Attention Network (propagate anomaly context across camera network)
       ↓
Network-Aware Anomaly Score (flag events that violate expected spatial propagation)
```

**What makes this novel:**
- Most traffic anomaly detection is **per-camera only**. Network-level anomaly detection (understanding how anomalies should propagate through a road network) is genuinely novel
- You're learning the **expected delay** between upstream and downstream cameras, then flagging when reality doesn't match
- This is the kind of spatial reasoning that impresses transportation researchers

**Feasibility on Colab T4:** ✅ Autoencoders are tiny. GAT on 87 nodes is trivial.

---

## The Novel Stack — Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   NOVEL CONTRIBUTION                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Weather-Conditioned Adaptive Detection              │
│     YOLOv11 + Edge Enhancement + Condition Routing      │
│     Novel: per-condition detection heads                │
│                                                         │
│  2. Graph-Based Cross-Camera Tracking                   │
│     BoT-SORT (per-cam) + GAT (cross-cam Re-ID)        │
│     Novel: spatial-temporal plausibility matching        │
│                                                         │
│  3. Weather-Traffic Cross-Modal Graph Transformer       │
│     GAT spatial + Transformer temporal + Cross-Attention│
│     Novel: multi-modal node features + cross-attention  │
│                                                         │
│  4. Network-Aware Multi-Scale Anomaly Detection         │
│     Autoencoder + GAT propagation                       │
│     Novel: anomaly propagation modeling across cameras   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Feasibility & Risk Assessment

| Component | Complexity | Risk | Fallback |
|---|---|---|---|
| Weather-adaptive detection | Medium | Low — incremental improvement on proven model | Even if improvement is small, benchmarks are publishable |
| Cross-camera Re-ID | High | High — may not work well with mixed resolutions | Document failures. Failures on real data are valuable. |
| Graph Transformer prediction | Medium | Low — 87-node graph trains fast | Start with vanilla LSTM baseline, add graph incrementally |
| Network-level anomaly | Medium | Medium — need enough data to learn propagation patterns | Fall back to per-camera anomaly if insufficient data |

> [!IMPORTANT]
> **The key insight:** You don't need ALL of these to be novel. Even **one** of these contributions, done rigorously with proper baselines and ablation studies, makes the project genuinely impressive. The others can use standard approaches. Pick the one that excites you most and go deep on it.

---

## What Training Will Actually Look Like

### Iterative Training Plan (Not Linear!)

```
Collect data (2 weeks)
     ↓
Train weather classifier → Evaluate → Tweak (2-3 iterations)
     ↓
Train YOLO baseline → Evaluate on Singapore → Find failure modes
     ↓
Design condition-specific heads based on failure analysis
     ↓
Train condition heads → Compare to baseline → Ablation study
     ↓
Run tracking → Extract trajectories → Build camera graph
     ↓
Train prediction model → Start simple (LSTM) → Add graph → Add cross-attention
     ↓
Train anomaly model → Verify against known anomaly patterns
     ↓
Write benchmarks and failure analysis
```

**Expect 3-5 training iterations per component.** Each iteration you'll discover something that changes your approach — that's research, not failure.

### Training Resource Estimates (All Colab T4)

| Component | Training Time | Iterations | Total GPU Hours |
|---|---|---|---|
| Weather classifier (ResNet-18) | ~30 min | 2-3 | ~1.5 hr |
| YOLO fine-tuning (per condition) | ~2-3 hr | 3-5 | ~12 hr |
| Cross-camera GAT | ~1 hr | 3-4 | ~4 hr |
| Graph Transformer prediction | ~2 hr | 4-5 | ~10 hr |
| Anomaly autoencoder + GAT | ~1 hr | 3-4 | ~4 hr |
| **Total** | | | **~32 hr** |

32 hours of Colab T4 — easily doable on the free tier over 1-2 weeks (12-hr session limit).

---

## Key Papers to Read Before Starting

| Priority | Paper | Why |
|---|---|---|
| 🔴 **Must read** | ECL-YOLOv11 (MDPI 2024) | Direct blueprint for weather-adaptive YOLO modifications |
| 🔴 **Must read** | STGformer (arXiv Oct 2024) | Best architecture for graph-based traffic prediction |
| 🔴 **Must read** | xMTrans (arXiv 2024) | Cross-modality Transformer for traffic — directly relevant |
| 🟡 **Should read** | MOTIP (CVPR 2025) | Understand SOTA tracking even if using BoT-SORT |
| 🟡 **Should read** | uTRAND (Apr 2024) | Graph-based trajectory anomaly detection |
| 🟡 **Should read** | DA-RAW (ECCV 2024) | Domain adaptation for adverse weather |
| 🟢 **Nice to have** | GE-STT (Feb 2025) | Graph-enhanced spatial-temporal transformer |
| 🟢 **Nice to have** | MTGAE (Jan 2024) | Mirror temporal graph autoencoder for anomaly |
