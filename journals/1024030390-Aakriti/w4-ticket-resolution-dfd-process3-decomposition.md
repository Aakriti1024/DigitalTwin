# Ticket 04: Hierarchical Data Flow Architecture & Process 3.0 (Agent Policy & Derate) Functional Decomposition

- **Owner:** Aakriti (1024030390)
- **Module:** Section 2 -- Data Flow Diagrams (Section 2.3, Figures 2.3, 2.4, 2.5)
- **Milestone:** Sprint 2 / Functional Data Flow Architecture

---

## 1. Problem Description & Difficulties Encountered

Data Flow Diagrams (DFDs) provide the functional perspective of software systems under classical structured analysis (Yourdon/DeMarco methodology). Decomposing our AI-driven digital twin into hierarchical DFDs presented significant technical hurdles:

1. **Confusion Between Control Flow vs. Data Flow:**
   - In deep learning systems, engineers instinctively draw flowchart arrows representing chronological execution (e.g., "Step 1: scale, Step 2: predict, Step 3: derate").
   - In DFD notation, directed arcs represent **data packets in motion**, circles represent **stateless transformations**, and parallel lines represent **persistent data stores**. Ensuring strict adherence to DFD syntax without embedding control logic was a primary challenge.
2. **Data Store Conservation Across Decomposition Levels:**
   - Academic guidelines require mathematical conservation between parent processes and child decompositions: any net input/output associated with a process at Level 1 must appear with identical net boundaries in Level 2.
   - In early drafts of Process 3.0 (`Agent Policy & Derate`), the Level 2 decomposition omitted the source data store `D2: RULTelemetry`, creating an unanchored data leak.
3. **Decomposing Algorithmic Policy Decision-Making:**
   - While physical data ingestion is straightforward, breaking down an autonomous prescriptive policy agent into deterministic sub-processes required cleanly separating threshold evaluation, policy matrix selection, and payload generation.

---

## 2. Technical Context & Hierarchical DFD Layers

Our system structures data flow across three distinct hierarchical tiers (Report Section 2.3):
- **Level 0 (Context Diagram, Figure 2.3):** Establishes the macro system boundary between external entities (**Turbofan Sensors**, **Maintenance Engineer / ERP**) and the core **Autonomous Agentic Digital Twin Control System**.
- **Level 1 (System Decomposition, Figure 2.4):** Modularizes the system into three primary processes:
  - `Process 1.0:` Scale & Window
  - `Process 2.0:` Predict RUL (PyTorch)
  - `Process 3.0:` Agent Policy & Derate
  Bridged by data stores `D1: SensorBuffer` and `D2: RULTelemetry`.
- **Level 2 (Process 3.0 Decomposition, Figure 2.5):** Focuses specifically on the internal mechanics of the autonomous decision engine.

```
DFD LEVEL 1 PIPELINE:
[Sensors] ---> (1.0 Scale & Window) ---> [D1: SensorBuffer]
                                                |
                                                v
                                      (2.0 Predict RUL) ---> [D2: RULTelemetry]
                                                                    |
                                                                    v
                                                     (3.0 Agent Policy & Derate) ---> [ECU / UI]

DFD LEVEL 2 DECOMPOSITION (PROCESS 3.0):
[D2: RULTelemetry] ---> (3.1 Evaluate Risk Thresholds)
                                    |
                          [Risk State / Health %]
                                    |
                                    v
                            (3.2 Select Derate Policy)
                                    |
                            [Selected Derate: 10%]
                                    |
                                    v
                            (3.3 Issue Control Payload) ---> [Engine Control Unit (ECU)]
```

---

## 3. Resolution & Engineering Implementation

### A. DFD Level 0 Context Diagram (Figure 2.3)
Formalized the single central process bubble (`Autonomous Agentic Digital Twin Control System`):
- **Incoming Flow:** `21-Channel Sensor Stream` from the external entity `Turbofan Sensors`.
- **Outgoing Flows:** `Derate Signals / Work Orders` directed to external entities `Maintenance Engineer / ERP` and the `Engine Control Unit (ECU)`.

### B. DFD Level 1 Modular System Decomposition (Figure 2.4)
Engineered the 3-process data transformation pipeline:
1. **Process 1.0 (Scale & Window):** Ingests raw tabular telemetry, applies standard scaling, constructs 30-cycle temporal tensors, and writes them into `D1: SensorBuffer`.
2. **Process 2.0 (Predict RUL - PyTorch):** Reads sequence arrays from `D1`, performs tensor forward propagation through the 2-layer LSTM network, and writes continuous cycle predictions into `D2: RULTelemetry`.
3. **Process 3.0 (Agent Policy & Derate):** Reads `D2`, applies risk threshold rules, and outputs control payloads to `ECU / Streamlit UI`.

### C. DFD Level 2 Decomposition of Process 3.0 (Figure 2.5)
I engineered the complete Level 2 functional decomposition of **Process 3.0 (Agent Policy & Derate)** into three clean, stateless transformations:
1. **Process 3.1: Evaluate Risk Thresholds:**
   - Ingests: `Predicted RUL` and engine identifier from `D2: RULTelemetry`.
   - Transformation: Compares RUL against configured boundary thresholds:
     - $\\text{RUL} > 60$: Normal / Nominal Condition
     - $35 < \\text{RUL} \\le 60$: Caution State
     - $15 < \\text{RUL} \\le 35$: Warning State (Action Required)
     - $\\text{RUL} \\le 15$: Critical State (Immediate Grounding)
   - Emits: Categorized `Risk State` data packet.
2. **Process 3.2: Select Derate Policy:**
   - Ingests: `Risk State` and current flight cycle index.
   - Transformation: Maps risk states to prescriptive control actions. For Warning state ($\\text{RUL} \\le 35$), selects a $10\\%$ thrust derating parameter ($0.90$ power rating factor).
   - Emits: `Selected Derate Policy` parameter.
3. **Process 3.3: Issue Control Payload:**
   - Ingests: `Selected Derate Policy` and engine telemetry metadata.
   - Transformation: Serializes the selected derating into a standardized JSON payload structure.
   - Emits: Physical control packets transmitted to the external `Engine Control Unit (ECU)` and dashboard visual alerts.

---

## 4. Verification & Outcome

- Achieved 100% data conservation between DFD Level 1 and Level 2: all inputs (`D2: RULTelemetry`) and outputs (`Engine Control Unit`) match exactly across hierarchy boundaries.
- Adheres strictly to Yourdon/DeMarco diagramming standards, eliminating illegal entity-to-entity and store-to-store connections.
- Formally integrated into Section 2.3 of the mid-semester evaluation report as Figures 2.3, 2.4, and 2.5.
