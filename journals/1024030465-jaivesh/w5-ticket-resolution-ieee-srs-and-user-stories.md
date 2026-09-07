# Ticket 05: IEEE-Format SRS Specification, Margin Overflows, and Story Card Modeling

- **Owner:** Jaivesh (1024030465)
- **Module:** Section 2 -- Software Requirement Specification in IEEE Format & User Stories
- **Milestone:** Sprint 2 / Requirements Engineering

---

## 1. Problem Description & Difficulty

Transforming an experimental machine learning script into an IEEE-compliant Software Engineering deliverable presented two major obstacles:
1. **LaTeX Table Margin Overflow:** Long textual descriptions in requirement tables and Agile story cards caused standard `tabular` columns to extend past the right page margin.
2. **Ambiguous Cyber-Physical Requirements:** Standard SRS templates assume CRUD applications. Specifying deterministic functional requirements for stochastic deep learning models and simulated degradation curves required rigorous requirement engineering.

---

## 2. Technical Context

When using standard LaTeX tables:
```latex
% Overflowed page margin because columns with 'p{...}' lacked dynamic ragged-right hyphenation
\begin{tabular}{|p{3cm}|p{12cm}|}
```
Furthermore, the Agile story cards required strict acceptance criteria that could be objectively verified during testing.

---

## 3. Resolution & Engineering Implementation

### A. Custom LaTeX Table Architecture
Implemented a dedicated column type with automatic ragged-right alignment and line breaks:
```latex
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
```
Replaced standard `tabular` with `longtable` environments spanning clean page breaks for:
- Use Case Specifications (UC-03, UC-04)
- User Stories Matrix (US-01 to US-06)
- Sample Story Card (US-05)

### B. Formal IEEE Requirements Mapping
Defined 9 Functional Requirements (FR-1 through FR-9) and 4 Non-Functional Requirements (NFR-1 through NFR-4):
- **FR-2 (Labeling):** Formally specified piece-wise linear RUL ground truth computation:
  $$\text{RUL}_i = \min(125.0, \text{Maximum Cycle} - \text{Current Cycle})$$
- **FR-8 & FR-9 (Digital Twin Simulation):** Required closed-loop comparison between intervention wear curves and passive baseline curves.
- **NFR-1 (Inference Latency):** Bounded single-engine prediction latency to $< 2.0\text{ seconds}$ on standard CPU hardware.

### C. Story Card US-05 Specification
Formalized acceptance criteria for the core novel feature:
```
Story ID: US-05 (Priority: High | 5 Story Points)
Story: As a Maintenance Engineer, I want to see the effect of an approved action 
       on the Digital Twin so that I can evaluate the resulting engine degradation 
       and time-on-wing extension.
Acceptance Criteria:
1. Given a recommended action, the engineer can approve the action.
2. The approved action is applied to the Digital Twin simulation.
3. The Digital Twin updates the simulated engine degradation trajectory.
4. The system compares the intervention trajectory with the passive baseline.
5. The system displays the estimated time-on-wing extension resulting from the approved action.
```

---

## 4. Verification & Outcome

- All SRS sections and user story tables compile with zero overfull `\hbox` warnings.
- Complete bi-directional traceability established between Functional Requirements (FR-1 to FR-9) and Agile User Stories (US-01 to US-06).
