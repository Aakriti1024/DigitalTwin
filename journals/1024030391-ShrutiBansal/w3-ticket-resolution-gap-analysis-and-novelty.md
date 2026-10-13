# Ticket 03: Literature Gap Analysis & Novelty Definition

- **Owner:** Shruti Bansal (1024030391)
- **Module:** Novelty and Project Positioning
- **Milestone:** Sprint 1 / Research & Scoping

---

## 1. Problem Description & Difficulties Encountered

Many projects built on C-MAPSS stop at predicting RUL. The task was to identify what existing work leaves out and turn that into a clear, defensible novelty statement. The observations were:

1. **Most systems stop at prediction:** they hand a static report to a human, with no closed-loop response to an intervention.
2. **An explicit recommend-and-approve step is rare:** very few systems model a step between prediction and action.
3. **Full autonomy is not acceptable in aviation:** safety requires a human decision-maker in the loop.

The difficulty was fitting all three observations into one coherent statement that is ambitious but still safe and buildable.

---

## 2. Technical Context

The gap sits between *predicting* a failure and *deciding what to do about it*. Closing that gap requires a prescriptive action, a way to measure its effect, and a governance step, which maps onto the three system layers from Ticket 01.

---

## 3. Resolution & Engineering Implementation

The novelty was finalized around **three pillars**:

1. **Predictive to Prescriptive shift:** the system prescribes a concrete operational action (design-stage example: de-rating engine thrust by 5-10%) to actively arrest degradation such as HPC wear, rather than only flagging it.
2. **Closed-loop digital twin simulation:** once an intervention is applied, the twin re-simulates the degradation trajectory and compares it against the passive run-to-failure baseline, producing a measurable cycles/hours-gained result.
3. **Explainable Human-in-the-Loop governance:** every recommendation carries an audit-ready rationale (design-stage example: *"RUL dropped below 35 cycles, so a 5% thrust de-rate is recommended to prevent premature blade creep"*), and the maintenance engineer keeps final approve/reject authority.

*Note: the percentages and thresholds above are design-stage illustrations. The thresholds and interventions actually implemented are defined in `config.py`.*

---

## 4. Verification & Outcome

- The three pillars are now the project's stated novelty in the README and report.
- Each pillar maps to a system component: the agent (prescription), the simulation module (closed loop) and the approval flow (HITL).
