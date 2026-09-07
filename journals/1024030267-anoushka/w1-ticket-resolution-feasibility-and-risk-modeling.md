# Ticket 01: Multi-Dimensional Feasibility Analysis, Scope Boundaries & Risk Assessment

- **Owner:** Anoushka (1024030267)
- **Module:** Section 1 -- Project Selection Phase, Feasibility Analysis & Risk Engineering
- **Report Reference:**
  - **Chapter 1, Section 1.1 (Feasibility Summary):** *Technical, Economic, Data, and Time Feasibility specifications.*
  - **Chapter 1, Section 1.1 (Risks):** *Evaluation of Model RMSE divergence across FD001–FD004, fixed-threshold sensitivity, simulated derate physics assumptions, and Human-in-the-Loop delayed approval behavior.*
  - **Chapter 1, Section 1.2 (Scope & Out of Scope):** *Demarcation of C-MAPSS FD001 focus vs. deferral of multi-condition subsets (FD002–FD004) and exclusion of live physical IoT sensors.*
- **Milestone:** Sprint 1 / Project Inception & Requirements Analysis

---

## 1. Problem Description & Difficulty

During the preliminary project proposal stage for **UCS503**, our initial project concept was overly ambitious. Early project drafts proposed integrating live IoT sensor hardware, training deep learning models across all four NASA C-MAPSS subsets (FD001 through FD004) simultaneously, and implementing certification-grade aviation safety compliance within a single semester.

During initial review, evaluators raised critical concerns regarding feasibility and scope creep:
1. **Multi-Condition Generalization Risk:** Sub-datasets FD002 and FD004 feature 6 complex operational flight conditions and multiple simultaneous failure modes (HPC and Fan degradation), leading to training instability and unpredictable convergence within standard lab computational budgets.
2. **Hardware Infeasibility:** Physical turbofan telemetry hardware and live engine testbeds are inaccessible in an academic lab setting.
3. **The Human-in-the-Loop (HITL) Operational Conundrum:** Aviation safety forbids unverified autonomous control. However, requiring human approval introduces an asynchronous operational risk: *What occurs if the Maintenance Engineer delays or fails to submit an approval? Does the system block, crash, or freeze?*

---

## 2. Technical Context & Root Cause Analysis

In software engineering project planning, ambiguous feasibility specifications and undefined boundary conditions lead to team desynchronization and milestone failure.

Initial feasibility drafts lacked structured evaluation criteria:

```latex
% EARLY FLAWED DRAFT: Subjective and unbounded feasibility claims
\subsection*{Feasibility}
The project is feasible because we can use Python and PyTorch which are free. 
We will download NASA datasets and train models on laptops. 
The system will run in real-time and predict engine health accurately.
```

Evaluators rejected this formulation because:
- It lacked quantitative computational constraints.
- It did not address runtime risks or hardware boundaries.
- It failed to account for what happens when human interaction is interrupted or absent.

---

## 3. Resolution & Engineering Implementation

### A. Four-Quadrant Feasibility Architecture (Report Section 1.1)
I restructured the feasibility analysis into four formal, mutually exclusive dimensions:
1. **Technical Feasibility:** Validated the software stack (Python 3.10+, PyTorch 2.0+, Streamlit, Pandas, NumPy, Scikit-Learn) against commodity CPU/GPU compute constraints.
2. **Economic Feasibility:** Verified that the entire pipeline leverages zero-cost, open-source permissive licenses (BSD/Apache/MIT), eliminating proprietary licensing or hardware procurement expenses.
3. **Data Feasibility:** Verified that the NASA C-MAPSS benchmark is publicly accessible, pre-labeled, and run-to-failure, completely eliminating sensor deployment risk and data collection delays.
4. **Time Feasibility:** Decoupled the project into four independent modules (**Data/Twin**, **Prognostics**, **Policy Agent**, and **UI/Integration**), allowing parallel development across all 4 team members within the 12-week semester timeline.

### B. In-Scope vs. Out-of-Scope Boundary Definition (Report Section 1.2)
To guarantee high-quality execution without scope creep, I formalized explicit boundaries:
- **In-Scope:** Focus on the benchmark **FD001** sub-dataset (single operating condition, single failure mode: High-Pressure Compressor degradation), offline model training, LSTM regression, rule-based prescriptive policy, and interactive dashboard mockups.
- **Out-of-Scope:** Deferring complex sub-datasets (FD002–FD004) to future stretch goals; strictly excluding physical jet engine hardware integration and certification-grade FAA/EASA airworthiness validation.

### C. Risk Assessment & Operational Behavior Modeling (Report Section 1.1)
I formulated four core technical risks with concrete engineering mitigations:

```latex
\subsection*{Risks}
\begin{itemize}
    \item Model accuracy (RMSE) may vary across the four C-MAPSS sub-datasets (FD001--FD004) 
          due to differing operating conditions and fault modes.
    \item Fixed-threshold policy logic may not generalize well if engine degradation 
          patterns deviate from those seen in training data.
    \item Simulated wear-rate reduction under derating is a modeling assumption, not 
          physically validated hardware data.
    \item Since the engineer must approve a recommended action before it is applied, 
          delayed or absent approval will leave the digital twin on the passive degradation 
          trajectory -- this is expected behaviour, not a system fault, but should be 
          clearly communicated during evaluation.
\end{itemize}
```

Notably, by formally specifying that **delayed or absent approval leaves the twin on its passive degradation baseline**, I eliminated deadlock states and established an unambiguous behavioral requirement for downstream development.

---

## 4. Verification & Outcome

1. **Academic Approval:** The formal Software Bid and Feasibility Summary passed mid-semester project proposal review by Dr. Stuti with zero revisions.
2. **Clear Team Division:** Establishing four decoupled modules enabled each team member to work independently without blocked dependencies.
3. **Robust Risk Posture:** Evaluators specifically commended the proactive analysis of human-in-the-loop delay risks, validating the engineering maturity of our project planning.
