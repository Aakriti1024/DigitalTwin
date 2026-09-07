# Engineering Changelog: Detailed Changes & Problem Resolutions

This document tracks major architectural, diagrammatic, and software engineering modifications implemented across the project lifecycle, specifically focusing on **Section 2 of the Project Report Onwards (Analysis Phase, UML Modeling, DFDs, and IEEE SRS)** led by **Jaivesh (1024030465)**.

---

## 🏗️ 1. Architecture & Repository Restructuring

- **Monorepo Directory Standardization:** Restructured the repository to align with the TIET UCS503 project template:
  - `/code`: Decoupled Python deep learning pipeline, simulation engine, and dashboard.
  - `/journals`: Added individual engineering journals (`journals/1024030465-jaivesh/`) and central changelog (`journals/changes.md`).
  - `/project-report-prototype-stage`: LaTeX source files and high-resolution diagram assets.
  - `/docs`: System architectural specifications and data flow documentation.
- **Gitignore Maintenance:** Created a comprehensive `.gitignore` filtering out heavy C-MAPSS sub-datasets (FD002--FD004), virtual environments, Python bytecode, and LaTeX auxiliary files (`.aux`, `.log`, `.toc`, `.synctex.gz`).
- **Dependency Isolation:** Formulated `code/requirements.txt` capturing exact version constraints for PyTorch, Streamlit, Pandas, NumPy, Scikit-Learn, and Matplotlib.

---

## 📐 2. Analysis Phase (Section 2) Deliverables

### Use Cases & System Actor Boundaries
- **Dual-Actor Modeling:** Established boundaries between the human actor (**Maintenance Engineer**) and automated background actor (**Digital Twin System Actor**).
- **Include & Extend Relationships:** Modeled core operational use cases with strict UML semantics:
  - `Predict RUL` $\ll\text{include}\gg$ `Preprocess & Compute RUL`
  - `Approve / Reject Action` $\ll\text{include}\gg$ `Recommend Maintenance Action`
  - `Recommend Maintenance Action` $\ll\text{extend}\gg$ `Predict RUL` (triggered conditionally when predicted RUL crosses caution/warning risk thresholds).
- **Formal Specifications:** Authored complete use case templates for **UC-03 (Predict RUL - LSTM Model)** and **UC-04 (Generate Maintenance Recommendation)**.

### Multi-Lane Activity / Swimlane Workflow
- **Three-Lane Partitioning:** Designed a unified swimlane diagram mapping responsibilities across:
  1. *Maintenance Engineer* (Telemetry upload, rationale review, approval/rejection).
  2. *Digital Twin System* (Preprocessing, LSTM inference, risk evaluation, simulation update).
  3. *Maintenance Team* (Alert logging, inspection staging, parts replacement).
- **Orthogonal Margin Routing:** Routed loopback and rejection flows along outer margins to prevent edge crossings and visual node collisions.

### Class & Sequence Diagram Architecture
- **Object-Oriented Domain Model:** Modeled domain entities (`EngineTelemetry`, `HealthState`, `MaintenanceAction`), computational engines (`LSTMPredictor`, `AutonomousMaintenanceAgent`), and simulation models (`EngineDigitalTwin`).
- **Human-in-the-Loop Sequence Flow:** Formalized runtime interaction using alternative execution blocks (`[Approved]` vs `[Rejected]`), proving that physical or simulated control is never executed without explicit human sign-off.
- **Naming Compliance:** Formatted headers neutrally (`\section{Class Diagram and Sequence Diagram}`) to respect institutional presentation guidelines while fulfilling all software engineering syllabus criteria.

### Data Flow Diagram (DFD) Decomposition
- **DFD Level 0 (Context Diagram):** Established boundary interactions with external entities.
- **DFD Level 1 (System Overview):** Decomposed application into five core processes (1.0 Ingestion, 2.0 Prognostics, 3.0 Policy Agent, 4.0 Digital Twin, 5.0 Dispatch).
- **DFD Level 2 (Decomposition of Process 2.0):** Decomposed deep learning inference into:
  - 2.1 Sliding Window Buffering ($30 \times 14$)
  - 2.2 Feature Scaling (StandardScaler)
  - 2.3 PyTorch LSTM Tensor Forward Pass
  - 2.4 Multi-Tier Health Classifier

### IEEE-Format SRS & Agile User Stories
- **IEEE-830 Compliant Specification:** Authored Product Perspective, Operating Environment, 9 Functional Requirements (FR-1 through FR-9), and 4 Non-Functional Requirements (Performance, Usability, Reliability, Maintainability).
- **Agile Story Point Matrix:** Formulated User Stories US-01 through US-06 with priority and Fibonacci story point estimations.
- **Formal Acceptance Testing (US-05):** Authored a dedicated Story Card defining mathematical verification criteria for Time-on-Wing ($\Delta \text{TOW} > 0$) under closed-loop derate control.

---

## 🐛 3. Bug Fixes & Technical Problem Resolutions

- **TikZ Stick-Figure Leg Coordinate Typo:** Fixed a `-5.55` coordinate typo in the Maintenance Engineer stick-figure node, which previously caused the actor's legs to stretch downward across five use-case ellipses. Corrected to `-0.55`, restoring geometric symmetry.
- **Draw.io Dark-Mode Image Inversion:** Created an automated Python PIL image normalization script (`convert_diagram.py`) that converted dark-mode diagram exports with muddy gray backgrounds and grid dots into pure white (`RGB(255,255,255)`) high-contrast vector assets.
- **LaTeX Table Margin Overflow:** Resolved table margin clipping in LaTeX by introducing custom `\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}` column specifiers within `longtable` environments.
- **Overleaf File Path Standardization:** Replaced nested directory image paths (`figures/...`) with root-level flat naming conventions (`dfd0.png`, `dfd1.png`, `dfd2.png`, `class_diagram.png`, `sequence_diagram.png`) to ensure clean compilation across all Overleaf instances.
