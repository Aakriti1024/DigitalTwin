# Engineering Changelog: Project Selection, IEEE SRS & Agile Requirements Engineering

This document tracks major software engineering contributions, analytical modeling, LaTeX architectures, and problem resolutions implemented across the project lifecycle, specifically focusing on **Chapter 1 (Project Selection Phase & SDLC Planning)** and **Chapter 2 (Sections 2.3 & 2.4: IEEE SRS Specification & Agile User Stories)** led by **Anoushka (1024030267)**.

All items directly document the formulation, structuring, and authoring of the deliverables presented in the **UCS 503 Mid-Semester Evaluation Project Report**.

---

## 🎯 1. Project Selection, Feasibility & Risk Engineering (Report Chapter 1, Section 1.1 & 1.2)

- **4-Quadrant Feasibility Framework:** Formulated the Technical, Economic, Data, and Time feasibility breakdown, establishing that the NASA C-MAPSS dataset, open-source stack (PyTorch, Streamlit, Pandas), and decoupled 4-module architecture make the project viable within the semester timeline.
- **In-Scope vs. Out-of-Scope Demarcation:** Defined strict project boundaries by centering the mid-semester scope on the single-condition **FD001** benchmark sub-dataset, while explicitly placing multi-condition subsets (FD002–FD004) and physical IoT engine hardware out of scope to avoid scope creep.
- **Human-in-the-Loop Delay Modeling:** Modeled the operational risk of delayed or absent engineer approval, explicitly establishing in the report that unapproved recommendations leave the digital twin on its passive baseline trajectory, preventing deadlock or error states.

---

## 🔄 2. Agile Scrum SDLC Architecture & TikZ Modeling (Report Chapter 1, Section 1.3)

- **TikZ Cyclic SDLC Workflow (Figure 1.1):** Designed and implemented the publication-quality vector diagram in LaTeX TikZ, resolving curved arrow collisions with the central feedback circle by engineering an orthogonal $2 \times 2$ grid with margin clearance.
- **Color-Coded Perimeter Routing:** Implemented dedicated palette styling (`blue!10`, `green!10`, `orange!10`, `purple!10`) with thick directional arrows ensuring high visual clarity across both digital PDF and print formats.
- **12-Week Sprint Schedule:** Decomposed the project into four 3-week sprint increments (Sprint 1: Data, Sprint 2: PyTorch LSTM, Sprint 3: Policy Engine, Sprint 4: UI & Integration) with balanced task allocations.

---

## 📋 3. IEEE-830 Software Requirement Specification (Report Chapter 2, Section 2.3)

- **Formal Functional Requirements (FR-1 through FR-9):** Re-authored ambiguous requirement drafts into 9 testable, RFC-2119 compliant functional requirements with mathematical RUL ground truth formulations ($\text{RUL}_i = \text{Maximum Cycle} - \text{Current Cycle}$).
- **Human Governance Safety Gate (FR-6 & FR-7):** Formally codified the requirement that prescriptive actions must present explainable rationale and require explicit human engineer approval before actuation.
- **Closed-Loop Verification Requirement (FR-8 & FR-9):** Enforced the mandatory comparative evaluation between intervention degradation trajectories and passive baselines to quantify Time-on-Wing ($\Delta \text{TOW}$) extensions.

---

## 📏 4. Non-Functional Requirements & Interface Contracts (Report Chapter 2, Section 2.3)

- **Quantifiable Quality Bounds (NFR-1 to NFR-4):** Established objective, testable thresholds:
  - NFR-1 (Performance): Single-engine RUL prediction latency bounded to $< 2.0\text{ seconds}$ on standard CPU hardware.
  - NFR-2 (Usability): Clear visual presentation of engine health, predictions, and alerts.
  - NFR-3 (Reliability): Multi-metric validation suite comprising MAE, RMSE, and $R^2$.
  - NFR-4 (Maintainability): Namespace decoupling across `data/`, `prognostics/`, `agent/`, and `dashboard/`.
- **System Boundary Contracts (Section 3.3):** Defined the CSV telemetry input schema, prediction output tuples, and the interactive Streamlit dashboard interface contract for upcoming implementation.

---

## 🃏 5. Agile User Stories & Story Card Formulation (Report Chapter 2, Section 2.4)

- **LaTeX Table Margin Overflow Resolution:** Replaced rigid `tabular` environments with custom ragged-right `longtable` column types (`\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}`), eliminating `Overfull \hbox` margin clipping.
- **User Story Matrix (US-01 through US-06):** Formulated user personas, capabilities, and business benefits, allocating Fibonacci story points ($3\text{ to }5\text{ pts}$, total velocity budget $= 24\text{ pts}$).
- **Story Card US-05 Acceptance Criteria:** Authored the formal 5-step testable acceptance criteria establishing the verification blueprint for upcoming closed-loop simulation and dashboard deliverables.
