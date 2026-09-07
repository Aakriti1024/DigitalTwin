# Ticket 03: Closed-Loop 3-Lane Swimlane Workflow & Asynchronous HITL Logic

- **Owner:** Aakriti (1024030390)
- **Module:** Section 2 -- Activity Diagram and Swimlane Diagrams (Section 2.2, Figure 2.2)
- **Milestone:** Sprint 1 / Behavioral Process Modeling

---

## 1. Problem Description & Difficulties Encountered

Section 2.2 of the Software Engineering report required modeling the end-to-end dynamic workflow of our system using an **Activity and Swimlane Diagram**. 

Unlike simple CRUD software where execution is strictly sequential, our agentic digital twin bridges physical engine telemetry with high-level enterprise logistics:
1. **Asynchronous Multi-Domain Partitioning:**
   - The workflow spans three distinct operational domains: human inspection (**Maintenance Engineer**), autonomous inference and simulation (**Autonomous Agentic Twin**), and external operational hardware/logistics (**Enterprise ERP / ECU**).
   - Partitioning responsibilities without introducing deadlocks or cross-lane ambiguities was a significant challenge.
2. **Modeling Non-Blocking Decision Branches:**
   - At the core of the system is the conditional check: *Is predicted RUL $\\le 35$ cycles?*
   - If False, the engine must continue nominal operation at 100% thrust rating without interruption.
   - If True, the agent must synchronously trigger a 10% thrust derate on the digital twin while asynchronously dispatching an ERP work order to reserve replacement parts.
   - Representing this split and synchronization without dangling activity nodes required careful behavioral modeling.
3. **Preventing Infinite Loops in Rejection Paths:**
   - In early draft iterations, when an engineer reviewed an alert or when RUL was above threshold, the feedback connector looped back to the start node rather than the streaming telemetry step, implying an erroneous complete system re-initialization.

---

## 2. Technical Context & Swimlane Partitions

To create an auditable engineering artifact, I established three explicit swimlanes corresponding to the system architecture:
- **Lane 1: Maintenance Engineer:** Primary human user interface. Handles simulation launch, engine selection, visual monitoring, and audit log inspection.
- **Lane 2: Autonomous Agentic Twin:** Computational core. Ingests telemetry, executes PyTorch LSTM forward pass, evaluates RUL against risk thresholds, and updates the physical wear simulation trajectory.
- **Lane 3: Enterprise ERP / ECU:** External actuation and enterprise logistics. Receives JSON work orders for maintenance bay staging and updates operational engine thrust setpoints.

```
+------------------------+-------------------------------+------------------------------+
|  MAINTENANCE ENGINEER  |    AUTONOMOUS AGENTIC TWIN    |     ENTERPRISE ERP / ECU     |
+------------------------+-------------------------------+------------------------------+
|                        |                               |                              |
| (O) Start              |                               |                              |
|  |                     |                               |                              |
|  v                     |                               |                              |
| [Select Engine &] ---->| [Stream Telemetry &]          |                              |
| [Launch Simulation]    | [Run PyTorch LSTM]            |                              |
|                        |  |                            |                              |
|                        |  v                            |                              |
|                        | [Compute Predicted RUL &]     |                              |
|                        | [Health %]                    |                              |
|                        |  |                            |                              |
|                        |  v                            |                              |
|                        | < RUL <= 35 Cycles? >         |                              |
|                        |   |             |             |                              |
|                        |   | (No)        | (Yes)       |                              |
|                        |   v             v             |                              |
|                        | [Continue]    [Command Closed-|                              |
|                        | [Nominal]     |Loop 10% Derate]                              |
|                        | [100% Rate]     |             |                              |
|                        |                 +------------>| [Dispatch ERP Work Order &]  |
|                        |                 |             | [Route Re-assignment]        |
|                        |                 v             |   |                          |
| [View Live Control] <--+-----------------+             |   |                          |
| [UI & Audit Log]       |                               |   |                          |
|  |                     |                               |   |                          |
|  +---------------------+-------------------------------+---+                          |
|  v                                                                                    |
| (X) End Node                                                                          |
+---------------------------------------------------------------------------------------+
```

---

## 3. Resolution & Engineering Implementation

### A. Formalizing Figure 2.2 Control Flow
I structured and finalized the 3-lane swimlane workflow (Report Figure 2.2):
1. **Initial Trigger:** Maintenance Engineer selects the target turbofan engine and initiates the simulation session.
2. **Inference Pipeline:** Control transitions to the **Autonomous Agentic Twin** lane, which streams the latest sensor sequence and executes the PyTorch LSTM neural model, outputting predicted RUL and engine health percentage.
3. **Conditional Threshold Evaluation:** The agent evaluates the decision node:
   - **Branch `[No]` (Nominal RUL $> 35$ cycles):** Transitions to `Continue Nominal Operation (100% Rating)` and routes directly to the telemetry visualization display.
   - **Branch `[Yes]` (Critical/Warning RUL $\\le 35$ cycles):** Executes `Command Closed-Loop 10% Thrust Derate`.
4. **Dual Dispatch & Actuation:**
   - Control bifurcates: a command payload is issued to the **Enterprise ERP / ECU** lane to `Dispatch ERP Work Order & Route Re-assignment`.
   - Concurrently, the twin recalculates the wear-rate trajectory under the 0.90 thrust derating.
5. **Synchronization & Final Review:** Both branches converge back to the Engineer lane at `View Live Control UI & Audit Log`, where the user inspects the historical intervention log before reaching the terminal end node.

### B. Resolution of Loopback Routing & Clean Termination
- Replaced ambiguous start-node loopbacks with dedicated state transitions: nominal operation simply updates the live control dashboard and waits for the next telemetry packet.
- Provided explicit termination semantics: the process safely reaches the double-ringed end node after all ERP dispatches and trajectory visual updates are rendered, ensuring compliance with standard UML activity modeling.

---

## 4. Verification & Outcome

- Fully validated against the operational logic implemented in the prototype codebase (`code/agent/policy.py` and `code/dashboard/app.py`).
- Clear separation between human monitoring and automated actuation established, satisfying academic evaluation requirements for closed-loop cyber-physical systems.
- Rendered cleanly as Figure 2.2 in Chapter 2 of the final project report.
