# Ticket 01: Cyber-Physical Scope Demarcation, 21-Sensor Telemetry Constraints & Feasibility Modeling

- **Owner:** Aakriti (1024030390)
- **Module:** Section 1 -- Software Bid & Project Overview (Sections 1.1 & 1.2)
- **Milestone:** Sprint 1 / Project Selection & Architectural Scoping

---

## 1. Problem Description & Difficulties Encountered

When initiating the **Autonomous Agentic Digital Twin for Predictive Maintenance and Closed-Loop Control of Turbofan Engines**, our primary software engineering hurdle in Section 1 (Project Selection Phase) was scoping the problem space to avoid both oversimplification and unfeasible project creep.

### Key Technical Challenges:
1. **Dataset Complexity & Dimensionality Explosion:**
   - The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark comprises four sub-datasets (FD001, FD002, FD003, FD004) characterized by varying operating flight conditions (sea level to 40,000 ft, Mach 0.0 to 0.84) and dual failure modes (HPC degradation and fan degradation).
   - Attempting to build an autonomous cyber-physical control loop across all multi-condition operating profiles simultaneously within a single-semester timeline introduced massive convergence uncertainty for the deep learning pipeline.
2. **Boundary Definition Between Cyber-Physical Twin vs. Hardware Deployment:**
   - In aerospace domains, confusion frequently arises regarding what constitutes the software digital twin versus the physical plant.
   - We faced difficulties defining an explicit boundary that satisfies rigorous Software Engineering academic evaluation while maintaining realistic industrial validity without requiring physical turbofan engine testbeds or avionics hardware-in-the-loop (HIL) test rigs.
3. **Formalizing Feasibility Across Conflicting Dimensions:**
   - Standard academic bids often treat feasibility superficially. We needed to construct a multi-dimensional feasibility evaluation (Technical, Economic, Time) that directly justified our architectural technology stack (PyTorch, Streamlit, Pandas, Scikit-Learn) and guaranteed deliverability across a 4-member engineering team.

---

## 2. Technical Context & Feasibility Formulation

Traditional predictive maintenance in academia operates strictly in an **offline, open-loop regime**: a model ingests sensor logs, outputs a Remaining Useful Life (RUL) number, and halts. 

In contrast, our project mandates an **Agentic Cyber-Physical System (CPS)** where the software twin dynamically interacts with a policy agent. Ingesting raw C-MAPSS telemetry directly into an unconstrained control loop posed severe risks:
- Unfiltered, constant sensor channels (e.g., fuel flow ratio, ambient pressure sensors that do not vary) inject noise into sequence models.
- Operational delay caused by required human engineer approvals could cause race conditions if the digital twin did not define clear fallback states.

---

## 3. Resolution & Engineering Implementation

### A. Strict Scope Partitioning & Benchmark Selection
To resolve the dimensionality and convergence risks, I established a formal demarcation in the Project Scope (Report Section 1.2):
- **Selected In-Scope Baseline:** Centered our prototype development on the **FD001** sub-dataset (single operating condition at sea level, high-pressure compressor failure mode), utilizing all **21 raw sensor channels** mapped into **30-cycle sliding window sequence buffers**.
- **Deferred Out-of-Scope:** Explicitly documented that multi-condition subsets (FD002–FD004), physical commercial engine integration, and FAA/EASA airworthiness certification are out-of-scope for the prototype stage.

```
+-------------------------------------------------------------------------+
|                        PROJECT SCOPE BOUNDARIES                         |
+-------------------------------------------------------------------------+
| IN-SCOPE:                                                               |
|  - NASA C-MAPSS FD001 (21 sensor channels, run-to-failure cycles)      |
|  - 30-cycle sliding window sequence buffering & StandardScaler scaling  |
|  - PyTorch sequence-to-one LSTM RUL regression                          |
|  - Closed-loop wear-rate simulation with empirical derate factor        |
|  - Autonomous Policy Agent with human approval gate                     |
|  - Streamlit interactive telemetry dashboard & JSON ERP dispatch API    |
+-------------------------------------------------------------------------+
| OUT-OF-SCOPE:                                                           |
|  - Physical jet engine hardware & testbed instrumentation               |
|  - Sub-datasets FD002-FD004 (multi-condition flight regimes)            |
|  - FAA / EASA DO-178C avionics flight software certification            |
+-------------------------------------------------------------------------+
```

### B. Formulating the 3-Tier Feasibility Matrix
I authored the formal Feasibility Analysis within the Software Bid (Report Section 1.1):
1. **Technical Feasibility:** Validated the viability of utilizing the NASA C-MAPSS run-to-failure repository paired with the PyTorch deep learning framework, Python-based closed-loop wear simulation models, and the Streamlit web framework.
2. **Economic Feasibility:** Structured the entire technology stack exclusively on open-source frameworks (Python 3.12, PyTorch 2.x, Streamlit, Pandas, NumPy, Scikit-Learn), ensuring $\\text{Cost} = \\$0.00$ with zero licensing or proprietary API dependencies.
3. **Time Feasibility:** Partitioned the software engineering lifecycle across four decoupled modules:
   - *Module 1:* Data Ingestion & Digital Twin Simulator
   - *Module 2:* Deep Learning Prognostics (PyTorch LSTM)
   - *Module 3:* Autonomous Maintenance Policy Agent
   - *Module 4:* Streamlit Dashboard & ERP Dispatch Gateway
   This enabled concurrent, sprint-based parallel execution across our 4-member group without blocking dependencies.

### C. Multi-Stakeholder Analysis Matrix
I identified and formalized the four primary system stakeholders and their functional touchpoints:
- **Maintenance Engineer:** Monitors real-time degradation curves and acts as the final authority on prescriptive control actions.
- **Fleet Operations Manager:** Uses predicted health indices for dynamic flight re-routing and maintenance schedule optimization.
- **Supply Chain / ERP System:** Ingests automated JSON payloads to pre-order replacement turbofan blades and schedule hangar bay allocation.
- **Data Science Team:** Continuously validates prediction accuracy, tracks model drift, and optimizes loss functions.

---

## 4. Verification & Outcome

- Successfully eliminated scope ambiguity, enabling the team to proceed directly into formal UML and DFD modeling without requirement churn.
- Established clear operational assumptions: if a human engineer is absent or delays approval, the digital twin automatically continues its baseline monitoring trajectory without crashing or entering undefined deadlocks.
- Formally approved and integrated into Chapter 1 (Sections 1.1 and 1.2) of the UCS 503 Mid-Semester Evaluation Project Report.
