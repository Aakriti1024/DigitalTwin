# Ticket 05: IEEE-830 SRS Specification & Agile Story Card US-03 Formal Acceptance Modeling

- **Owner:** Aakriti (1024030390)
- **Module:** Section 2 -- Software Requirement Specification in IEEE Format & Agile User Stories (Sections 2.4 & 2.5)
- **Milestone:** Sprint 2 / Requirements Engineering & Acceptance Modeling

---

## 1. Problem Description & Difficulties Encountered

In academic software engineering projects involving machine learning, a common flaw is treating requirements informally (e.g., "the model should predict accurately"). 

For Section 2.4 (IEEE SRS) and Section 2.5 (User Stories & Story Cards), our challenge was to formulate **deterministic, verifiable, and RFC-2119 compliant specifications** for a cyber-physical deep learning system:
1. **Specifying Physics-Informed Requirements:**
   - How do you write a testable functional requirement for a wear simulation that responds dynamically to an AI derating command?
   - A vague requirement like "wear should decrease" cannot be tested. We needed an explicit mathematical relationship governing how the derate factor modifies the physical degradation curve.
2. **Establishing Realistic, Quantifiable Non-Functional Requirements (NFRs):**
   - We needed to specify latency, accuracy, usability, and maintainability metrics that could be empirically validated on standard CPU development environments without requiring enterprise cloud clusters.
3. **Agile Story Card Acceptance Criteria Formulation (US-03):**
   - User stories are high-level user desires, but story cards must provide unambiguous acceptance criteria. Authoring the acceptance criteria for **Story Card US-03** (Autonomous Thrust Derate by Fleet Manager) required defining clear, testable preconditions, trigger actions, and observable outcomes.

---

## 2. Technical Context & IEEE-830 Standards

Under IEEE Std 830-1998, requirements must be correct, unambiguous, complete, consistent, verifiable, and traceable. 

To bridge machine learning inference with cyber-physical simulation, I mapped the mathematical degradation model into formal requirements:
- Baseline turbofan wear rate: $w_0(t) = \\alpha \\cdot t^\\beta$
- Derated wear rate under $10\\%$ thrust reduction ($D = 0.90$):
  $$w_{\\text{derate}}(t) = w_0(t) \\times (D)^{2.5} = w_0(t) \\times (0.90)^{2.5} \\approx 0.768 \\times w_0(t)$$
This physics-based formulation results in an approximate $23.2\\%$ reduction in instantaneous mechanical wear, providing a deterministic mathematical baseline for testing requirement compliance.

---

## 3. Resolution & Engineering Implementation

### A. Codifying Functional Requirements (FR-1 through FR-7)
I authored the 7 core Functional Requirements in Section 2.4 (IEEE SRS):
- **FR-1:** The system shall accept tabular C-MAPSS sensor telemetry files (FD001–FD004).
- **FR-2:** The system shall normalize features using `StandardScaler` and form 30-cycle sliding sequence windows ($30 \\times 21$).
- **FR-3:** The PyTorch LSTM model shall output continuous RUL regression values for each valid sequence window.
- **FR-4:** The Agent Policy Engine shall evaluate predicted RUL against defined multi-tier thresholds: Caution ($60\\text{ cycles}$), Warning ($35\\text{ cycles}$), and Critical ($15\\text{ cycles}$).
- **FR-5:** The system shall apply a $10\\%$ thrust derating factor ($0.90$ power rating) when predicted RUL drops $\\le 35\\text{ cycles}$.
- **FR-6:** The system shall recalculate the wear rate reduction using the non-linear relationship ($\\text{Derate}^{2.5}$) to update health state trajectories in real time.
- **FR-7:** The Streamlit dashboard shall render side-by-side comparative plots comparing the unmitigated baseline trajectory against the agentic digital twin trajectory.

### B. Defining Measurable Non-Functional Requirements (NFR-1 through NFR-4)
I established four objective quality bounds:
- **NFR-1 (Performance):** RUL inference per 30-cycle sequence window shall complete in $< 50\\text{ milliseconds}$ on standard CPU hardware.
- **NFR-2 (Accuracy):** The PyTorch LSTM model shall achieve a Root Mean Square Error $\\text{RMSE} \\le 11.0\\text{ cycles}$ on benchmark test sets.
- **NFR-3 (Usability):** The dashboard shall feature intuitive visual status cards (Green/Yellow/Red) and searchable JSON audit logs.
- **NFR-4 (Modularity):** The codebase shall enforce a four-layer namespace separation (`digital_twin`, `prognostics`, `agent`, `dashboard`).

### C. Agile User Stories Matrix & Story Card US-03
I authored the prioritized Agile User Story Matrix (Report Section 2.5):

| ID | User Story | Priority | Story Points |
| :--- | :--- | :--- | :--- |
| **US-01** | As a Maintenance Engineer, I want to view live sensor telemetry so that I can monitor engine state. | High | 3 |
| **US-02** | As a Maintenance Engineer, I want PyTorch LSTM RUL predictions so that I know remaining engine life. | High | 5 |
| **US-03** | As a Fleet Manager, I want the AI Agent to automatically derate thrust when wear is high so that engine failure is prevented. | High | 8 |
| **US-04** | As a Maintenance Engineer, I want automated JSON work orders generated so that repair parts are ordered in advance. | Medium | 3 |
| **US-05** | As a Data Scientist, I want side-by-side trajectory plots comparing baseline vs. agentic performance so that I can verify life extension. | Medium | 5 |

#### Detailed Story Card: US-03
```
+---------------------------------------------------------------------------------------+
| STORY CARD: US-03                                                                     |
+---------------------------------------------------------------------------------------+
| Story:                                                                                |
|   As a Fleet Manager, I want the AI Agent to automatically derate thrust when wear    |
|   is high so that engine failure is prevented.                                        |
|                                                                                       |
| Acceptance Criteria:                                                                  |
|   1. When predicted RUL drops below 35 cycles, the agent commands a 10% thrust derate |
|      (0.90 power setting).                                                            |
|   2. The derating factor reduces physical wear rate in the simulation model using the  |
|      Derate^2.5 non-linear formula.                                                   |
|   3. The dashboard displays a green (agentic) vs. red (baseline) side-by-side life    |
|      extension trajectory curve.                                                      |
|                                                                                       |
| Priority: High                                                                        |
| Estimate: 8 Story Points                                                              |
+---------------------------------------------------------------------------------------+
```

---

## 4. Verification & Outcome

- Complete bi-directional traceability established between Functional Requirements (FR-1 to FR-7) and Agile User Stories (US-01 to US-05).
- All acceptance criteria for Story Card US-03 were verified via automated simulation unit tests in `code/digital_twin/simulator.py`.
- Formally approved and incorporated into Sections 2.4 and 2.5 of the UCS 503 Mid-Semester Evaluation Project Report.
