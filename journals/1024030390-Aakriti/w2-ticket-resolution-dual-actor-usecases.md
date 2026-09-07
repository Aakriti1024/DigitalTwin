# Ticket 02: Dual-Actor Governance & Autonomous Agentic Use Case Modeling

- **Owner:** Aakriti (1024030390)
- **Module:** Section 2 -- Use Cases (Section 2.1, Figure 2.1, UC-01, UC-02)
- **Milestone:** Sprint 1 / Behavioral Modeling & System Actor Specification

---

## 1. Problem Description & Difficulties Encountered

In classical software engineering, UML Use Case diagrams represent software systems as passive rectangular containers where human actors trigger external actions (e.g., login, submit form, view report). 

However, in our **Autonomous Agentic Digital Twin**, the architecture operates as a cyber-physical system with an active background entity:
1. **The System as an Autonomous Actor:**
   - The AI agent is not merely a database or library; it continuously ingests high-frequency telemetry, runs neural inferences, evaluates risk cost matrices, and executes autonomous control commands.
   - We encountered difficulty modeling whether the automated decision agent should be drawn inside the system boundary as a subsystem or outside the system boundary as a **System Actor**.
2. **Ambiguity in UML Dependency Stereotypes (`<<include>>` vs. `<<extend>>`):**
   - Team debates arose regarding how to model the relationship between RUL prediction, risk evaluation, and physical thrust derating.
   - A naive model drew direct associations from actors to all use cases without capturing execution prerequisites, violating UML 2.5 standards and obscuring the conditional nature of prescriptive interventions.
3. **Handling Variable-Length Sequences in UC-01:**
   - In the NASA C-MAPSS dataset, early operational test sequences may have fewer cycles than our sliding window length ($W = 30$). Documenting how the system behaves prior to cycle 30 required formulating rigorous alternate flows in the use case templates.

---

## 2. Technical Context & Actor Boundaries

Under aviation safety guidelines, automated control systems require clear separation between human-initiated supervision and autonomous background routines:
- **Maintenance Engineer (Human User):** Initiates session, reviews visual degradation curves, examines explainable rationale, and oversees dispatch operations.
- **Autonomous Policy Agent (System Actor):** Autonomous daemon executing the OODA (Observe-Orient-Decide-Act) loop, calculating real-time risk scores, and issuing control signals.

```
       +-----------------------+                    +------------------------------------+
       |  MaintenanceEngineer  |                    | Autonomous PolicyAgent (Sys Actor) |
       |     [Human User]      |                    |         [Background Daemon]        |
       +-----------+-----------+                    +-----------------+------------------+
                   |                                                  |
                   |  (Uploads / Monitors)                            |  (Ingests / Predicts)
                   v                                                  v
     +------------------------------------------------------------------------------------+
     |                         DIGITAL TWIN SYSTEM BOUNDARY                               |
     |                                                                                    |
     |  [Stream Sensor Telemetry] <-----+ <<include>>                                     |
     |                                  |                                                 |
     |                      [Predict RUL (LSTM Model)] <----+ <<include>>                 |
     |                                                      |                             |
     |                                          [Evaluate Policy & Risk]                  |
     |                                                      ^                             |
     |                                                      | . . . <<extend>>            |
     |                                                      | (Condition: RUL <= 35)      |
     |                                          [Execute Thrust Derating]                 |
     |                                                                                    |
     |  [View Live Control UI]                  [Dispatch ERP Work Order]                 |
     +------------------------------------------------------------------------------------+
```

---

## 3. Resolution & Engineering Implementation

### A. Architectural Formalization of Figure 2.1
To resolve the actor boundary and dependency ambiguities:
- Placed the **Maintenance Engineer** on the left boundary as the primary human operator interacting with `Stream Sensor Telemetry`, `View Live Control UI`, and approving prescriptive actions.
- Placed the **Autonomous Policy Agent** on the right boundary as a dedicated **System Actor**, formally linked to `Stream Sensor Telemetry`, `Predict RUL (LSTM Model)`, `Evaluate Policy & Risk`, `Execute Thrust Derating`, and `Dispatch ERP Work Order`.
- Codified strict dependency relationships:
  - `Predict RUL (LSTM Model)` $\\ll\\text{include}\\gg$ `Stream Sensor Telemetry`: Sensor streaming and feature extraction are mandatory prerequisites for model inference.
  - `Evaluate Policy & Risk` $\\ll\\text{include}\\gg$ `Predict RUL (LSTM Model)`: Risk evaluation cannot execute without a fresh RUL regression output.
  - `Execute Thrust Derating` $\\ll\\text{extend}\\gg$ `Evaluate Policy & Risk`: Explicitly modeled as an extension point activated **only** when the predicted RUL drops into the Warning or Critical zone ($\\text{RUL} \\le 35\\text{ cycles}$). Under nominal conditions, the base use case completes without triggering the derate extension.

### B. Authoring Detailed Use Case Specifications
I authored the complete tabular Use Case specifications for **UC-01** and **UC-02** in Section 2.1.2:

#### 1. UC-01: Predict RUL & Engine State (PyTorch LSTM)
- **Use Case ID:** UC-01
- **Actor(s):** Autonomous Policy Agent (System Actor)
- **Description:** System ingests a sliding window sequence of 21 sensor channels and outputs an estimated Remaining Useful Life (RUL) value.
- **Preconditions:** Sensor telemetry is scaled; PyTorch LSTM model weights are loaded in memory.
- **Main Flow:**
  1. System extracts 30-cycle sliding window sequence.
  2. Sequence is normalized using `StandardScaler`.
  3. PyTorch LSTM model executes forward inference pass.
  4. Predicted RUL and computed health index are forwarded to the Policy Engine.
- **Alternate Flow (Buffering Mode):** If current sequence length is $< 30$ cycles, the system enters buffering mode, accumulating cycles until the 30-cycle minimum window is satisfied.
- **Postconditions:** RUL prediction is available for policy decision-making and dashboard telemetry.

#### 2. UC-02: Execute Closed-Loop Thrust Derating
- **Use Case ID:** UC-02
- **Actor(s):** Autonomous Policy Agent
- **Description:** When predicted RUL drops below configured risk thresholds, the agent issues closed-loop thrust derating commands to extend engine lifespan.
- **Preconditions:** Predicted RUL is updated (UC-01).
- **Main Flow:**
  1. Agent checks predicted RUL against thresholds (Caution: 60, Warning: 35, Critical: 15).
  2. If $\\text{RUL} \\le 35$, Agent selects `DERATE 10%` (0.90 rating).
  3. Agent sends control payload to Engine Control Unit (ECU).
  4. Simulator applies derate factor, reducing physical wear rate for subsequent cycles.
  5. System logs action and updates health trajectory curves.
- **Postconditions:** Thermal stress is reduced and physical wear rate is decelerated.

---

## 4. Verification & Outcome

- The use-case model accurately reflects the cyber-physical duality of our architecture, fulfilling all software engineering syllabus criteria.
- The inclusion of alternate flows in UC-01 eliminated edge-case crashes during initial startup when fewer than 30 cycles are available.
- Incorporated into Chapter 2 (Section 2.1) of the official report and fully verified against the system behavioral specifications.
