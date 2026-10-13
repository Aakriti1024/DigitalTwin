# Ticket 04: System Workflow, Functional Mapping & Design Diagrams

- **Owner:** Shruti Bansal (1024030391)
- **Module:** System Design: Workflow, Use Case, Activity and DFD Levels 0-2
- **Milestone:** Sprint 1 / Analysis & Design

---

## 1. Problem Description & Difficulties Encountered

With the novelty defined, it had to become something concrete enough to build. The challenges were:

1. **Turning a high-level statement into parts:** mapping the interactions between the Prediction, Twin and Governance layers.
2. **Defining the decision logic:** deciding exactly how a threshold check leads to an intervention proposal.
3. **Showing before-and-after:** deciding how the intervention-vs-baseline comparison is generated and presented to the engineer.

---

## 2. Technical Context

The workflow was written out first, then used to derive the diagrams. It traces:
- how sensor data enters and moves through the prediction layer
- how a threshold check decides whether an intervention should be proposed
- how an approved intervention triggers a re-simulation in the twin layer
- how the comparison is generated and shown to the engineer

---

## 3. Resolution & Engineering Implementation

The following diagrams were prepared from the finalized scope:

| Diagram | What it models |
| :--- | :--- |
| **Use Case Diagram** | Actors (Maintenance Engineer, Digital Twin / Prediction System) and use cases: view engine health, request RUL prediction, receive recommendation, approve/reject it, view intervention-vs-baseline comparison. |
| **Activity Diagram** | End-to-end flow: sensor ingestion, RUL prediction, threshold check, recommendation, engineer decision, conditional re-simulation. |
| **DFD Level 0** | System boundary: sensor / C-MAPSS data feed and Maintenance Engineer as external entities around one "Digital Twin PdM System" process. |
| **DFD Level 1** | Sub-processes: data preprocessing, RUL prediction, threshold/decision check, intervention simulation, rationale generation, with data stores for raw sensor logs and historical trajectories. |
| **DFD Level 2** | Finer decomposition of RUL prediction and intervention simulation: feature extraction, normalization, model inference, baseline retrieval, trajectory re-computation. |

---

## 4. Verification & Outcome

- The diagrams gave a clear view of what each layer must do and became the reference for implementation.
- The workflow confirmed that no intervention is simulated without an approval step, supporting the HITL requirement.
