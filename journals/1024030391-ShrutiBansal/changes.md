# Engineering Changelog: Research, Dataset Scoping, Novelty, Workflow Design & ML Pipeline

This document tracks the contributions of **Shruti Bansal (1024030391)** across the project lifecycle: domain research, dataset benchmarking, novelty definition, workflow and diagram design, and the data preprocessing / model training pipeline.

---

## 🔍 1. Domain Research & System Layering

- **Digital Twin & PHM Foundations:** Studied digital twin architecture, PHM, RUL estimation, predictive vs. prescriptive maintenance, closed-loop simulation and HITL governance.
- **Turbofan Internals:** Studied engine modules and how degradation such as HPC wear appears in temperature, pressure, rotor speed and fuel flow signals.
- **Three-Layer Architecture:** Defined the Prediction, Simulation/Twin and Governance layers.

## 📊 2. Dataset & Benchmark Scoping

- **C-MAPSS Study:** Analysed FD001-FD004 and justified **FD001** (single condition, single HPC fault) as the starting scope.
- **Label & Evaluation Study:** Studied RUL derivation from run-to-failure trajectories and literature evaluation practice.

## 💡 3. Novelty Definition

- **Gap Analysis:** Identified that existing C-MAPSS work stops at prediction.
- **Three Pillars:** Predictive-to-Prescriptive shift, closed-loop twin simulation, explainable HITL governance.

## 📐 4. Workflow & Design Diagrams

- **Workflow Mapping:** Traced data flow from ingestion through prediction, threshold check, approval and re-simulation.
- **Diagrams:** Use Case, Activity, DFD Level 0, DFD Level 1, DFD Level 2.

## ⚙️ 5. Data & Model Pipeline (in progress)

- **[FILL]** Summarize your preprocessing and training changes once Ticket 05 is complete.

---

## 🐛 6. Bug Fixes & Technical Problem Resolutions

- **[FILL]** List real problems you fixed (for example: data path issues, model score problems, shape mismatches), with the cause and the fix.
