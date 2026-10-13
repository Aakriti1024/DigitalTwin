# Ticket 02: NASA C-MAPSS Dataset Study & Benchmark Scoping

- **Owner:** Shruti Bansal (1024030391)
- **Module:** Dataset Selection and Benchmarking
- **Milestone:** Sprint 1 / Research & Scoping

---

## 1. Problem Description & Difficulties Encountered

The project needed a dataset that is credible and comparable to prior work. NASA's **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) dataset is the standard benchmark for turbofan prognostics, but it ships as four sub-datasets (FD001-FD004) with different complexity. The challenges were:

1. **Choosing the right scope:** Deciding which sub-dataset to start with without overcomplicating a first working system.
2. **Understanding the labels:** C-MAPSS contains run-to-failure sensor trajectories, so RUL labels have to be derived rather than read directly.
3. **Knowing how to compare results:** Understanding how existing literature evaluates and scores models on this dataset.

---

## 2. Technical Context

- **FD001-FD004:** four sub-datasets differing in operating conditions and fault modes.
- **FD001:** a single operating condition and a single fault mode (HPC degradation), which is the simplest and cleanest starting point.
- **RUL derivation:** each engine runs until failure, so the remaining life at any cycle is derived from the engine's total lifetime in the trajectory.

---

## 3. Resolution & Engineering Implementation

### A. Scope Decision: FD001
FD001 was selected as the starting scope because its single operating condition and single HPC fault mode match the "isolate one well-understood fault" decision from Ticket 01.

### B. Label and Evaluation Understanding
- Studied how RUL labels are derived from run-to-failure trajectories.
- Studied how published work evaluates and scores models on C-MAPSS so that our results can be compared fairly against it.

---

## 4. Verification & Outcome

- FD001 adopted as the project's benchmark scope.
- This understanding feeds the preprocessing and training pipeline (see Ticket 05).
