# Autonomous Agentic Digital Twin for Predictive Maintenance and Closed-Loop Control of Turbofan Engines

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Course](https://img.shields.io/badge/Course-UCS503%20Software%20Engineering-green.svg)]()
[![Group](https://img.shields.io/badge/Group-3C25-orange.svg)]()

> **UCS 503 Software Engineering Lab Project**  
> Computer Science and Engineering Department, Thapar Institute of Engineering & Technology (TIET), Patiala.  
> **Evaluation:** Mid-Semester Evaluation | **Submitted to:** Dr. Stuti

---

## 👥 Team Details (Group 3C25)

| S.No. | Roll Number | Name | Branch / Year | Core Responsibility |
| :---: | :---: | :---: | :---: | :--- |
| 1 | 1024030390 | Aakriti | BE CoE, 3rd Year | Project Lead & Section 1 Planning / SDLC |
| 2 | 1024030391 | Shruti | BE CoE, 3rd Year | Data Preprocessing & Model Training |
| 3 | 1024030465 | Jaivesh | BE CoE, 3rd Year | Section 2 Analysis Phase, UML/DFD Architecture & IEEE SRS |
| 4 | 1024030267 | Anoushka | BE CoE, 3rd Year | UI / Streamlit Integration & Dashboard |

---

## 🚀 Project Overview & Novelty

Most predictive maintenance projects stop at **passive prediction**: an AI model estimates Remaining Useful Life (RUL) as a raw scalar and leaves interpretation entirely to human guesswork. 

This project bridges that gap by engineering a **prescriptive, closed-loop Cyber-Physical Digital Twin**:
1. **Prescriptive Action Policy:** When predicted RUL drops below configured risk thresholds (Caution, Warning, Critical), a rule-based agent automatically prescribes a mitigating control action (e.g., graded thrust derate: 5%, 10%, or emergency shutdown) accompanied by an **explainable rationale**.
2. **Human-in-the-Loop (HITL) Safety Gate:** Aviation systems cannot permit unverified autonomous control. The Maintenance Engineer reviews the proposed action and rationale on an interactive dashboard, retaining 100% override/approval authority.
3. **Closed-Loop Wear Simulation:** Once approved, the action is injected into the engine's Digital Twin simulation, updating its degradation trajectory and proving measurable **Time-on-Wing (TOW)** extension compared to passive run-to-failure baseline.

---

## 🏗️ Repository Architecture & File Guide

This repository is structured following institutional Software Engineering monorepo guidelines:

```
DigitalTwin/
├── code/                                # Complete implementation codebase
│   ├── agent/
│   │   ├── __init__.py
│   │   └── policy.py                   # Rule-based prescriptive policy agent & risk thresholds
│   ├── dashboard/
│   │   └── app.py                      # Interactive Streamlit control center
│   ├── digital_twin/
│   │   ├── __init__.py
│   │   └── simulator.py                # Aero-thermodynamic degradation twin & derate physics
│   ├── prognostics/
│   │   ├── __init__.py
│   │   └── model.py                    # PyTorch 2-layer LSTM RUL regression model
│   ├── data/                           # Benchmark telemetry datasets (C-MAPSS FD001)
│   ├── generate_plots.py               # Evaluation plotting scripts (RUL, wear curves)
│   ├── train_and_simulate.py           # End-to-end training and closed-loop simulation pipeline
│   └── requirements.txt                # Python package dependencies
├── docs/                               # Architectural and analytical documentation
│   └── architecture.md                 # System overview and pipeline mechanics
├── journals/                            # Individual student engineering journals & changelogs
│   ├── 1024030465-jaivesh/             # Jaivesh's contribution journal (Section 2 onwards)
│   │   ├── index.md                    # Overview of contributions & milestone index
│   │   ├── w1-ticket-resolution-activity-swimlane.md
│   │   ├── w2-ticket-resolution-dfd-multi-level-decomposition.md
│   │   ├── w3-ticket-resolution-usecase-actor-geometry-bug.md
│   │   ├── w4-ticket-resolution-uml-class-and-sequence-integration.md
│   │   └── w5-ticket-resolution-ieee-srs-and-user-stories.md
│   └── changes.md                      # Detailed technical changelog & resolution history
├── project-report-prototype-stage/     # LaTeX academic evaluation report
│   ├── report.tex                      # Complete LaTeX source file (UCS 503 Report)
│   ├── dfd0.png                        # Clean DFD Level 0 Context Diagram
│   ├── dfd1.png                        # Clean DFD Level 1 Modular System Diagram
│   ├── dfd2.png                        # Clean DFD Level 2 Detailed Process 2.0 Decomposition
│   ├── use_case_diagram.png            # Rendered Use-Case Diagram
│   └── README.md                       # Compilation instructions for Overleaf / LaTeX
├── .gitignore                          # Ignored caches, large files, and LaTeX artifacts
└── README.md                           # Master project documentation
```

---

## 📊 Section 2 (Analysis Phase) Architectural Highlights

The Analysis Phase (authored by **Jaivesh**) establishes the software engineering integrity of the project:
* **Use Cases:** Two primary actors (**Maintenance Engineer** & **Digital Twin System Actor**) with formal `<<include>>` and `<<extend>>` associations.
* **Activity & Swimlane Diagram:** Models the asynchronous flow across Maintenance Engineer, Digital Twin System, and Ground Maintenance Team.
* **Class & Sequence Diagrams:** Structural object model decoupled across Domain Entities, Computational Engines, and Simulation Controllers; sequence diagram explicitly models the `alt [Approved / Rejected]` execution frames.
* **Hierarchical DFDs:** Level 0 (Context), Level 1 (5 Subsystems), and Level 2 (Decomposition of Process 2.0: Window Ingest $\to$ Normalization $\to$ LSTM Inference $\to$ Health Classification).
* **IEEE SRS & User Stories:** 9 Functional Requirements (FR-1 to FR-9), 4 Non-Functional Requirements, and 6 Agile User Stories with story point estimation.

---

## ⚡ Quick Start & Execution

### 1. Prerequisites
Ensure Python 3.10+ is installed:
```bash
git clone https://github.com/Aakriti1024/DigitalTwin.git
cd DigitalTwin/code
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the LSTM & Run Closed-Loop Simulation
```bash
python train_and_simulate.py
```
This trains the PyTorch LSTM model on NASA C-MAPSS FD001, runs an autonomous closed-loop simulation on a sample test unit, and outputs trajectory logs to `results/agentic_digital_twin_simulation.csv`.

### 3. Launch the Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
The dashboard allows real-time telemetry streaming, manual parameter injection, and Human-in-the-Loop recommendation approval.

---

## 📝 LaTeX Report Compilation
The complete report is available in `/project-report-prototype-stage/report.tex`. It is formatted for direct compilation on Overleaf using `pdflatex`. Diagrams (`dfd0.png`, `dfd1.png`, `dfd2.png`) are referenced directly at the root level of the report directory.
