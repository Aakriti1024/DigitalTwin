# Ticket 03: IEEE-830 Functional Requirements Engineering for Cyber-Physical AI

- **Owner:** Anoushka (1024030267)
- **Module:** Section 2 -- Software Requirement Specification (SRS) in IEEE Format
- **Report Reference:**
  - **Chapter 2, Section 2.3 (Section 3.1 Functional Requirements):** *Formal specification of requirements FR-1 through FR-9.*
  - **Chapter 2, Section 2.3 (Section 1.1 Purpose & 1.2 Scope):** *Introduction and product boundaries.*
- **Milestone:** Sprint 2 / Requirements Engineering & Formal Specification

---

## 1. Problem Description & Difficulty

Standard Software Requirement Specification (SRS) frameworks (specifically **IEEE Std 830-1998**) were conceived for deterministic, transactional business applications (e.g., banking systems, inventory management, e-commerce). Specifying formal functional requirements for a **cyber-physical artificial intelligence system**—which couples stochastic deep learning regression with numerical aero-thermodynamic degradation simulations—presents substantial software engineering challenges:

1. **Non-Verifiable Ambiguity:** Early drafts frequently lapsed into subjective language (e.g., *"The model should predict RUL accurately"*, *"The system should optimize engine lifespan"*). Such statements fail IEEE-830 requirements verifiability criteria because they lack testable pass/fail conditions.
2. **Mathematical Labeling Ambiguity:** Specifying how training ground truth is derived from raw turbofan operational cycles without ambiguous English phrasing.
3. **Decoupling Autonomous Suggestion from Actuation:** The SRS had to strictly formalize the architectural boundary where the algorithm's authority terminates (`FR-5`, `FR-6`) and the human engineer's authority begins (`FR-7`), ensuring that the system cannot autonomously mutate the Digital Twin state (`FR-8`) without verified authorization.

---

## 2. Technical Context & Root Cause Analysis

In IEEE-830 standards, every functional requirement must be:
- **Unambiguous:** Having only one semantic interpretation.
- **Verifiable:** Providing a finite, objective process to prove compliance.
- **Traceable:** Mapped directly to upstream user needs and downstream test cases.

Early requirement formulations violated these principles:

```latex
% REJECTED EARLY DRAFT: Ambiguous, un-testable requirement statements
\begin{itemize}
    \item System should load jet engine datasets and make them clean.
    \item System must predict remaining life with good accuracy.
    \item The AI should recommend what the engineer should do to save the engine.
    \item Simulation will show that the engine lasts longer after the action.
\end{itemize}
```

These statements could not be converted into automated test assertions or formal software validation protocols.

---

## 3. Resolution & Engineering Implementation

I re-engineered the functional requirements using strict **RFC 2119 keyword semantics ("shall")**, embedding explicit mathematical definitions and operational boundaries across the pipeline:

### A. The 9 Formal Functional Requirements (Report Section 3.1)

```latex
\subsubsection*{3.1 Functional Requirements}
\begin{itemize}
    \item \textbf{FR-1:} The system shall accept turbofan engine sensor data in CSV format 
          compatible with the NASA C-MAPSS dataset.
    \item \textbf{FR-2:} The system shall preprocess sensor data and generate RUL labels 
          for training using $\text{RUL}_i = \text{Maximum Cycle} - \text{Current Cycle}$.
    \item \textbf{FR-3:} The system shall predict RUL for a given engine sequence using 
          the trained LSTM model.
    \item \textbf{FR-4:} The system shall classify the predicted RUL into predefined 
          health conditions.
    \item \textbf{FR-5:} The system shall generate a maintenance recommendation based on 
          the predicted RUL and defined risk thresholds.
    \item \textbf{FR-6:} The system shall display the predicted RUL, health status, 
          recommended action, and rationale on the dashboard.
    \item \textbf{FR-7:} The system shall allow the Maintenance Engineer to approve or 
          reject a recommended action.
    \item \textbf{FR-8:} The system shall apply an approved action to the Digital Twin 
          and update the simulated engine degradation trajectory.
    \item \textbf{FR-9:} The system shall compare the intervention and passive degradation 
          trajectories to estimate time-on-wing extension.
\end{itemize}
```

### B. Architectural Rigor & Pipeline Traceability
1. **Mathematical Ground-Truth (FR-2):** Formally defined the piecewise ground-truth target formulation, ensuring the data engineering team and machine learning team aligned on identical regression targets.
2. **The Prescriptive Bridge (FR-4 & FR-5):** Decomposed prediction into classification and action: an RUL number is first classified into an operational health condition (`Normal`, `Caution`, `Warning`, `Critical`), which then triggers an associated mitigating action policy.
3. **The Human Safety Barrier (FR-6 & FR-7):** Explicitly required human-readable explainability (`"rationale"`) alongside the recommendation, establishing the legal and operational basis for the engineer's approval or rejection.
4. **Closed-Loop Verification (FR-8 & FR-9):** Made the comparative evaluation between the intervention trajectory and the passive baseline a mandatory software requirement, enabling quantitative verification of the project's primary thesis ($\Delta \text{TOW} > 0$).

---

## 4. Verification & Outcome

1. **100% IEEE-830 Compliance:** All 9 requirements passed faculty review with zero ambiguity flags.
2. **Bi-Directional Traceability:** Established a clean 1-to-1 mapping connecting:
   - Use Cases (`UC-03` $\to$ `FR-2, FR-3`; `UC-04` $\to$ `FR-4, FR-5, FR-6, FR-7, FR-8, FR-9`).
   - Downstream Agile User Stories (`US-01` through `US-05`).
3. **Deterministic Testing Blueprint:** Each requirement was structured so that a simple unit test or integration test can verify its completion (e.g., asserting that `FR-8` cannot fire if `FR-7` returns `False`).
