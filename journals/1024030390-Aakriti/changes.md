# Engineering Changelog: Project Selection, System Scoping, DFDs & IEEE SRS

This document tracks major software engineering contributions, analytical modeling, data flow decompositions, and technical problem resolutions implemented across the project lifecycle, specifically focusing on **Chapter 1 (Project Selection Phase & System Overview)** and **Chapter 2 (Analysis Phase: Dual-Actor Use Cases, Swimlane Control, DFD Process 3.0 Decomposition, IEEE SRS & Story Card US-03)** led by **Aakriti (1024030390)**.

All items directly document the formulation, structuring, and authoring of the deliverables presented in the **UCS 503 Mid-Semester Evaluation Project Report**.

---

## 🎯 1. Project Selection, System Scoping & Feasibility (Report Chapter 1, Section 1.1 & 1.2)

- **Problem Domain & Motivation Formulation:** Formulated the foundational problem domain within Aerospace Cyber-Physical Systems (CPS) and Industrial IoT. Established the paradigm shift from traditional passive RUL regression to an active, prescriptive agentic system where an autonomous policy agent dynamically evaluates real-time telemetry and recommends closed-loop operational actions.
- **Scope Demarcation (21 Sensor Channels & 30-Cycle Windows):** Constrained the project intake pipeline to the NASA C-MAPSS FD001 run-to-failure benchmark, establishing that all 21 raw sensor channels must be normalized using `StandardScaler` and partitioned into 30-cycle temporal sliding windows ($30 \times 21$).
- **Strict In-Scope vs. Out-of-Scope Demarcation:** Formally established that physical engine hardware integration, live avionics flight testing, and FAA/EASA certification are out of scope, protecting the team from scope creep while keeping the software digital twin scientifically grounded.
- **3-Tier Feasibility Matrix:** Authored the Technical, Economic, and Time feasibility breakdown, proving that an open-source software stack (PyTorch, Streamlit, Pandas, NumPy, Scikit-Learn) with zero proprietary licensing costs allows rapid parallel development across four decoupled modules.
- **Multi-Stakeholder Architecture:** Mapped functional responsibilities across four distinct stakeholder personas: *Maintenance Engineer*, *Fleet Operations Manager*, *Supply Chain / ERP System*, and *Data Science Team*.

---

## 📐 2. Analysis Phase (Report Chapter 2) Deliverables

### Dual-Actor Use Cases & Template Specifications (Section 2.1)
- **Dual-Actor Modeling:** Established the operational boundaries between the human actor (**Maintenance Engineer**) and the automated background actor (**Autonomous Policy Agent**).
- **Include & Extend Semantics:** Modeled core operational interactions with strict UML stereotypes:
  - `Predict RUL (LSTM Model)` $\\ll\\text{include}\\gg$ `Stream Sensor Telemetry`
  - `Evaluate Policy & Risk` $\\ll\\text{include}\\gg$ `Predict RUL (LSTM Model)`
  - `Execute Thrust Derating` $\\ll\\text{extend}\\gg$ `Evaluate Policy & Risk` (triggered conditionally when predicted RUL breaches the Warning threshold of $\\le 35$ cycles).
- **Formal Use Case Specifications:** Authored detailed, complete use case specifications:
  - **UC-01:** *Predict RUL & Engine State (PyTorch LSTM)* detailing preconditions, 4-step main flow, alternate buffering flow for sequences $< 30$ cycles, and postconditions.
  - **UC-02:** *Execute Closed-Loop Thrust Derating* detailing automated threshold evaluation, ECU control payload generation, wear-rate reduction calculation, and postconditions.

### Closed-Loop Activity & Swimlane Workflow (Section 2.2)
- **Three-Lane Architectural Partitioning:** Designed and formalized the 3-lane swimlane control workflow:
  1. *Maintenance Engineer* (Launches simulation, reviews telemetry, evaluates proposed derating, and views live audit logs).
  2. *Autonomous Agentic Twin* (Ingests telemetry, executes PyTorch LSTM forward pass, evaluates RUL against Caution/Warning/Critical thresholds, commands closed-loop 10% derate).
  3. *Enterprise ERP / ECU* (Receives automated work-order dispatch payloads and executes turbine thrust re-calibration).
- **Asynchronous Decision Branching:** Modeled the core closed-loop decision: if $\\text{RUL} \\le 35$ cycles, trigger a 10% thrust derate recommendation; if $\\text{RUL} > 35$, continue nominal operation at 100% rating without interrupting the flight profile.

### Hierarchical Data Flow Architecture & Process 3.0 Decomposition (Section 2.3)
- **DFD Level 0 (Context Diagram):** Defined the external boundary interfaces exchanging raw 21-channel sensor streams, derate control signals, and work orders with Turbofan Sensors, the Maintenance Engineer, and the Enterprise ERP.
- **DFD Level 1 (System Decomposition):** Decomposed the application into three modular pipeline processes:
  - `Process 1.0:` Scale & Window (populating data store `D1: SensorBuffer`)
  - `Process 2.0:` Predict RUL via PyTorch (writing continuous estimates to `D2: RULTelemetry`)
  - `Process 3.0:` Agent Policy & Derate (ingesting predictions and emitting ECU payloads and UI telemetry)
- **DFD Level 2 (Decomposition of Process 3.0 -- Agent Policy & Derate):** Engineered the granular internal data flows of the autonomous policy agent:
  - `Process 3.1:` Evaluate Risk Thresholds (comparing RUL against Caution 60, Warning 35, Critical 15)
  - `Process 3.2:` Select Derate Policy (selecting 10% thrust derate / 0.90 rating)
  - `Process 3.3:` Issue Control Payload (formatting JSON control packets for the Engine Control Unit)

### IEEE-830 Software Requirement Specification (Section 2.4)
- **7 Functional Requirements (FR-1 through FR-7):** Formally codified requirements covering C-MAPSS ingestion, 30-cycle sliding window buffering, continuous RUL regression, multi-tier risk thresholds (Caution 60, Warning 35, Critical 15), automated 10% thrust derating, non-linear wear rate recalculation ($\\text{Derate}^{2.5}$), and side-by-side Streamlit comparative trajectory rendering.
- **4 Non-Functional Requirements (NFR-1 through NFR-4):** Established quantifiable bounds:
  - *NFR-1 (Performance):* RUL inference latency bounded to $< 50\\text{ ms}$ on CPU.
  - *NFR-2 (Accuracy):* PyTorch LSTM model constrained to achieve $\\text{RMSE} \\le 11.0$ cycles on test benchmarks.
  - *NFR-3 (Usability):* Streamlit dashboard featuring intuitive visual status cards and JSON audit logs.
  - *NFR-4 (Modularity):* Decoupled architecture across data, prognostics, agent, and dashboard layers.

### Agile User Stories & Story Card US-03 (Section 2.5)
- **Agile User Story Matrix (US-01 through US-05):** Authored 5 prioritized user stories spanning Maintenance Engineer, Fleet Manager, and Data Scientist personas with Fibonacci story point sizing ($3\\text{ to }8\\text{ pts}$).
- **Formal Story Card US-03:** Authored the dedicated story card for Fleet Manager autonomous thrust derating, establishing three testable acceptance criteria for closed-loop wear reduction and side-by-side visual curve verification.

---

## 🐛 3. Bug Fixes & Technical Problem Resolutions

- **C-MAPSS Data Ingestion Sequence Length Bug:** Resolved a data pipeline crash where test engines with fewer than 30 operating cycles triggered index out-of-bounds errors in the sequence generator. Engineered a zero-padding and minimum sequence buffering fallback flow documented in UC-01 alternate flows.
- **Agent Policy Infinite Decision Loop Resolution:** Fixed a structural flaw in initial swimlane control drafts where unapproved derate actions had no terminal route, causing an infinite loop. Routed rejected actions back to nominal monitoring with an explicit audit log.
- **DFD Process 3.0 Data Store Conservation Fix:** Corrected a Level 1 to Level 2 DFD inconsistency where control signals were depicted leaving Process 3.0 without sourcing input telemetry from data store `D2: RULTelemetry`. Restored strict data conservation across all decomposition levels.
- **Non-Linear Wear Equation Calibration:** Replaced an unrealistic linear wear model with an empirical aerodynamic degradation power law ($\\text{WearRate} \\propto \\text{Derate}^{2.5}$), preventing negative or flatlined RUL curves during closed-loop simulation.
- **IEEE SRS Margin Alignment:** Standardized requirement tables to prevent horizontal text clipping across standard A4 document boundaries.
