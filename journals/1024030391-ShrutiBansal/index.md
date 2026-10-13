# Engineering Journal: Research Scoping, Dataset Benchmarking, Novelty Definition, Workflow Design & Data/Model Pipeline

- **Student Name:** Shruti Bansal
- **Roll Number:** 1024030391
- **Group:** 3C25 (BE CoE, 3rd Year)
- **Course:** UCS503 - Software Engineering Lab
- **Guide:** Dr. Stuti Chug
- **Institute:** Thapar Institute of Engineering and Technology, Patiala
- **Assigned Project Scope:** **Domain Research & Novelty Definition, Dataset Study (C-MAPSS), System Workflow & Diagram Preparation, and Data Preprocessing & Model Training**

---

## Weekly Ticket & Milestone Index

| Ticket / Week | Focus Area & Project Reference | Core Technical Challenge | Status |
| :--- | :--- | :--- | :--- |
| [**Ticket 01**](./w1-ticket-resolution-digital-twin-and-turbofan-research.md) | **Digital Twin, PHM & Turbofan Internals**<br>*(Ref: Domain research, system layers)* | Understanding what a digital twin must model for a turbofan, and structuring the system into Prediction, Twin and Governance layers. | **Resolved** |
| [**Ticket 02**](./w2-ticket-resolution-cmapss-dataset-and-benchmarking.md) | **C-MAPSS Dataset & Benchmarking**<br>*(Ref: FD001-FD004, RUL labels)* | Choosing the right sub-dataset scope and understanding how RUL labels are derived from run-to-failure data. | **Resolved** |
| [**Ticket 03**](./w3-ticket-resolution-gap-analysis-and-novelty.md) | **Literature Gap & Novelty Definition**<br>*(Ref: Project novelty, three pillars)* | Converting an observed literature gap into a concrete, defensible novelty statement for a safety-critical domain. | **Resolved** |
| [**Ticket 04**](./w4-ticket-resolution-workflow-and-design-diagrams.md) | **System Workflow, Requirements & Diagrams**<br>*(Ref: Use case, Activity, DFD L0-L2)* | Turning the novelty statement into buildable parts and modelling behaviour with UML and data-flow diagrams. | **Resolved** |
| [**Ticket 05**](./w5-ticket-resolution-preprocessing-and-model-training.md) | **Data Preprocessing & RUL Model Training**<br>*(Ref: `ml/preprocess.py`, `ml/models.py`, `ml/train.py`)* | Building the FD001 pipeline and training the RUL prediction model. | **In Progress** |

---

## Scope & Contribution Summary

1. **Domain Research (Ticket 01):** Studied digital twin architecture (physical asset, virtual model, data link), PHM, RUL estimation, predictive vs. prescriptive maintenance, closed-loop simulation and human-in-the-loop governance. Studied turbofan internals and how degradation (e.g., HPC wear) appears in sensor signals.
2. **Dataset & Benchmarking (Ticket 02):** Analysed the NASA C-MAPSS FD001-FD004 structure and justified FD001 (single operating condition, single HPC fault mode) as the starting scope.
3. **Novelty Definition (Ticket 03):** Reviewed existing literature, identified that most systems stop at prediction, and finalized three pillars: Predictive-to-Prescriptive shift, closed-loop twin simulation, and explainable HITL governance.
4. **Workflow & Diagrams (Ticket 04):** Defined the end-to-end workflow and prepared the Use Case, Activity, and DFD Level 0/1/2 diagrams.
5. **Data & Model Pipeline (Ticket 05):** Implementation phase covering the C-MAPSS FD001 preprocessing pipeline and RUL model training.

## Next Steps

- Finish evaluating and tuning the RUL model
- Connect the trained model to the digital twin layer
- Build and refine the intervention-vs-baseline comparison
- Support the rationale-generation and approval interface
