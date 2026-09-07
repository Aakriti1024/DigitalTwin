# Ticket 02: Multi-Level DFD Hierarchical Decomposition and Image Asset Optimization

- **Owner:** Jaivesh (1024030465)
- **Module:** Section 2 -- Data Flow Diagrams (DFD Level 0, Level 1, Level 2)
- **Milestone:** Sprint 2 / Functional Decomposition

---

## 1. Problem Description & Difficulty

Our system integrates raw sensor telemetry with deep learning sequence models and cyber-physical simulations. Representing this in Data Flow Diagrams (DFD) presented two major challenges:

1. **Semantic Multi-Level Decomposition:**
   - Academic evaluators required hierarchical decomposition adhering to Edward Yourdon / DeMarco notation.
   - **Level 0 (Context):** Needs to establish the system boundary with external entities (Turbofan Engine, Maintenance Engineer, Enterprise Maintenance System).
   - **Level 1 (System):** Must partition the application into modular processes (Ingestion, RUL Prognostics, Policy Recommendation, Digital Twin Simulation, Alert Dispatch) while maintaining strict data store conservation.
   - **Level 2 (Detailed):** Must decompose the complex deep learning core (Process 2.0 -- Predict RUL) into its mathematical components (window buffering, normalization, tensor inference, and threshold classification).
2. **Visual Inversion & Asset Formatting:**
   - Initial diagrams designed in web tools (Draw.io) had dark backgrounds (`RGB(18,18,18)`) and grid dots.
   - Exported PNGs looked unreadable and unprofessional when embedded into LaTeX reports and printed in grayscale/black-and-white.

---

## 2. Technical Context

When exporting high-resolution diagrams from dark-mode canvases, simple color inversion creates inverted line artifacts and muddy mid-tones:

```python
# Naive inversion inverted black text to light gray with ugly gray grid patterns
from PIL import Image, ImageOps
inverted = ImageOps.invert(img) # produced gray background with visible grid dots
```

---

## 3. Resolution & Engineering Implementation

### A. Python Image Processing Pipeline
I engineered an automated PIL-based thresholding and contrast-normalization script (`convert_diagram.py`):
1. **Dynamic Dynamic-Range Expansion:** Mapped dark canvas background pixels ($< 40$) to pure white ($255, 255, 255$) while preserving black vectors and typography.
2. **Grid-Dot Removal:** Filtered faint grid-dot artifacts using adaptive thresholding.
3. **Automated Bounding-Box Cropping:** Trimmed unnecessary canvas margins to maximize diagram scale when embedded via `\includegraphics[width=\textwidth]{...}`.

```python
# Custom contrast expansion and whitening logic
from PIL import Image
import numpy as np

def whiten_diagram(input_path, output_path, low_thresh=40, high_thresh=180):
    img = Image.open(input_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    
    # Invert and normalize contrast
    arr = 255.0 - arr
    arr = np.clip((arr - low_thresh) * (255.0 / (high_thresh - low_thresh)), 0, 255)
    
    out_img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    # Auto-crop to content
    bbox = out_img.getbbox()
    if bbox:
        out_img = out_img.crop(bbox)
    out_img.save(output_path)
```

### B. Hierarchical DFD Structure
1. **DFD Level 0 (Context Diagram):**
   - Inputs: Raw sensor data streams from physical engines.
   - Outputs: Action approvals from Maintenance Engineer; work orders to Maintenance Depot; telemetry logs to Data Store.
2. **DFD Level 1 (Architecture):**
   - Decomposed into 5 discrete processes: Ingestion (1.0), RUL Prognostics (2.0), Prescriptive Policy Agent (3.0), Digital Twin Simulator (4.0), and Notification/Dispatch (5.0).
3. **DFD Level 2 (Decomposition of Process 2.0):**
   - Focused entirely on the AI pipeline:
     - Subprocess 2.1: Sliding Window Sequencer (packs continuous stream into $30 \times 14$ tensors).
     - Subprocess 2.2: MinMax Normalizer (scales inputs against C-MAPSS FD001 statistics).
     - Subprocess 2.3: PyTorch LSTM Inference (computes scalar regression RUL).
     - Subprocess 2.4: Degradation Health Classifier (assigns Normal, Caution, Warning, or Critical state).

---

## 4. Verification & Outcome

- All three DFD figures were rendered with pure white backgrounds, razor-sharp vector lines, and uniform font readability.
- Passed evaluation review with zero data-conservation mismatches between Level 0, Level 1, and Level 2.
