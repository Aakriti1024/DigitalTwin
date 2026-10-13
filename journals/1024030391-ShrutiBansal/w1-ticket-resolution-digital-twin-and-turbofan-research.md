# Ticket 01: Digital Twin Architecture, PHM Foundations & Turbofan Engine Internals

- **Owner:** Shruti Bansal (1024030391)
- **Module:** Domain Research and System Layering
- **Milestone:** Sprint 1 / Research & Scoping

---

## 1. Problem Description & Difficulties Encountered

The project aims to mirror a physical engine's health in near real time and help decide the right maintenance action *before* failure. Before any design could begin, two knowledge gaps had to be closed:

1. **What a digital twin actually is in this context:** Digital twin systems combine a physical asset, a virtual model and a data link between them. It was necessary to understand how existing predictive maintenance systems are structured and where they fall short when it comes to *acting* on their own predictions.
2. **What a turbofan twin needs to model:** Without understanding the mechanical structure of the engine, it is not possible to decide which sensors and fault modes matter.

---

## 2. Technical Context

**Concepts studied:**
- Digital twin architecture: physical asset, virtual model, and the data link between them
- Prognostics and Health Management (PHM) approaches
- Remaining Useful Life (RUL) estimation
- Predictive vs. prescriptive maintenance
- Closed-loop simulation vs. static reporting dashboards
- Human-in-the-loop (HITL) governance in safety-critical systems

**Turbofan architecture:** fan, low- and high-pressure compressors, combustor, high- and low-pressure turbines, and nozzle, and how each stage contributes to thrust generation.

**Health-monitoring signals:** temperature, pressure, rotor speed and fuel flow, and how degradation (for example, high-pressure compressor wear) shows up in them.

---

## 3. Resolution & Engineering Implementation

### A. Three-Layer System Structure
The research was consolidated into three layers, each handling one stage of the maintenance decision process:

| Layer | Responsibility |
| :--- | :--- |
| **Prediction Layer** | Estimates RUL from sensor data. |
| **Simulation / Twin Layer** | Re-simulates the degradation trajectory under a proposed intervention and compares it to the passive baseline. |
| **Governance Layer** | Generates a rationale for the recommendation and routes it to a human maintenance engineer for approval. |

### B. Scoping Decision
Isolating a single, well-understood fault mode was chosen as the right starting scope for a first working system, rather than modelling every possible failure.

---

## 4. Verification & Outcome

- The three-layer structure became the basis for the project's architecture and diagrams.
- The single-fault-mode scoping decision directly informed the dataset choice (see Ticket 02).
