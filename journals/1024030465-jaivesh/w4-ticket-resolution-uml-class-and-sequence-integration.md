# Ticket 04: System Domain Modeling (Class Diagram) and Dynamic Message Sequencing

- **Owner:** Jaivesh (1024030465)
- **Module:** Section 2 -- Class Diagram and Sequence Diagram
- **Milestone:** Sprint 2 / Object-Oriented Analysis & Design

---

## 1. Problem Description & Difficulty

Software engineering lab specifications required adding Class and Sequence diagrams to complement functional use cases. However, two structural problems arose:
1. **Institutional Naming Constraint:** The instructor explicitly requested that the term `"UML"` should not appear in the section titles or headings, requiring neutral, professional structural titles while fulfilling syllabus criteria.
2. **HITL Runtime Modeling:** Modeling the dynamic sequence of events where an automated deep learning prediction triggers an action recommendation, but the execution branches conditionally based on human engineer approval.

---

## 2. Technical Context

In classical automated systems, the model output directly commands physical actuators. In aviation predictive maintenance, this violates airworthiness standards. The sequence diagram needed to formally depict the **Human-in-the-Loop** barrier:

```
[Telemetry] -> Gateway -> LSTM Predictor -> Policy Engine -> Engineer Dashboard
                                                                    |
                                                            [Review & Approve?]
                                                                    |
                                                    +---------------+---------------+
                                                    | [Approved]                    | [Rejected]
                                                    v                               v
                                          Apply Derate to Twin             Log Override & Monitor
```

---

## 3. Resolution & Engineering Implementation

### A. Document Organization
- Structured the section neutrally as:
  ```latex
  \section{Class Diagram and Sequence Diagram}
  \subsection{Class Diagram}
  ...
  \subsection{Sequence Diagram}
  ...
  ```
- Positioned strategically between high-level behavioral flow (Activity/Swimlane Diagram) and detailed data pipelines (Data Flow Diagrams).

### B. Class Diagram Domain Architecture
Modeled the core object-oriented components across three layers:
1. **Domain Entities:** `EngineTelemetry`, `HealthState`, `MaintenanceAction`.
2. **Computational Services:** `LSTMPredictor` (`predict_rul()`, `load_weights()`), `PolicyAgent` (`evaluate_risk()`, `generate_recommendation()`).
3. **Simulation & UI:** `EngineDigitalTwin` (`apply_derate()`, `step_degradation()`), `DashboardController`.

### C. Sequence Diagram Alternative Execution
Modeled the message sequence using an `alt` interaction frame:
- **Condition `[Approved]`:** Engineer clicks approval $\to$ Dashboard sends `apply_action(derate_factor)` to `DigitalTwinSimulator` $	o$ Simulation updates wear rate $	o$ Dispatch API logs work order.
- **Condition `[Rejected]`:** Engineer rejects proposal $	o$ Dashboard logs engineer rationale $	o$ Engine remains on baseline monitoring trajectory.

---

## 4. Verification & Outcome

- Created clear, decoupled software boundaries between inference algorithms and interactive UI controls.
- Referenced flat image assets (`class_diagram.png`, `sequence_diagram.png`) ready for seamless inclusion in Overleaf.
