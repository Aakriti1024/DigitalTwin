# Ticket 01: Multi-Lane Asynchronous Decisions in TikZ Swimlane Activity Diagrams

- **Owner:** Jaivesh (1024030465)
- **Module:** Section 2 -- Activity Diagram and Swimlane Diagrams
- **Milestone:** Sprint 1 / Analysis Phase Architecture

---

## 1. Problem Description & Difficulty

In modeling the closed-loop operational cycle of the Turbofan Digital Twin, the workflow spans three distinct entities:
1. **Maintenance Engineer** (Human operator performing evaluation and approval)
2. **Digital Twin System** (Automated background actor performing telemetry ingest, inference, and simulation)
3. **Ground Maintenance Team** (Field personnel receiving staged work orders and executing physical maintenance)

### Key Technical Hurdles:
1. **Geometric Layout & Horizontal Overflow:** Fitting three distinct swimlanes with decision diamonds and activities inside standard A4 LaTeX page margins ($1\text{ in}$ margin) without clipping labels.
2. **Overlapping Feedback Loops:** The process has two crucial feedback mechanisms:
   - When RUL is normal ($> \text{Threshold}$), the system loops back to the telemetry ingestion phase.
   - When the engineer rejects the recommendation, the system branches to continue monitoring rather than triggering physical maintenance.
   Standard straight-arrow connectors crossed directly through existing activity boxes, causing visual collisions and unreadable diagrams.

---

## 2. Technical Context

Original attempt using standard TikZ node placement caused severe overlapping:

```latex
% Unconstrained routing caused arrows to pierce through intermediary boxes
\draw[flow] (threshold.west) -- (upload.west);
\draw[flow] (reject.west) -- (threshold.north);
```

Furthermore, standard decision diamonds with `aspect=1.0` caused multi-line text (`Review Proposed Derate & Explainable Rationale`) to breach diamond boundaries.

---

## 3. Resolution & Engineering Implementation

To resolve this, I established a parameterized geometric grid in TikZ:
1. **Dimension Standardization:**
   - Total width: $16.5\text{ cm}$ partitioned into three equal $5.5\text{ cm}$ vertical swimlanes.
   - Height: $19.5\text{ cm}$ with dedicated lane headers styled with `fill=gray!6`.
2. **Aspect Ratio Optimization:** Configured decision nodes with `aspect=2.0`, `minimum width=2.8cm`, and `minimum height=1.0cm`, allowing multi-line decision queries to fit with ample padding.
3. **Orthogonal Margin Routing:** Implemented waypoint routing along the outer left perimeter ($x = 0.75\text{ cm}$) to route loopback flows safely away from activity nodes:

```latex
% Normal RUL bypass: channeled along left outer margin
\draw[flow] (threshold.west) -- node[pos=0.4, above, font=\scriptsize, fill=white, inner sep=1.5pt] {No (Normal RUL)} 
    (0.75,-7.6) -- (0.75,-3.2) -- (upload.west);

% Rejection branch: routed back to monitoring loop
\draw[flow] (reject.west) -- (0.75,-15.6) -- (0.75,-7.6);
```

4. **Multi-Lane Crossing:** Used right-angle routing (`-|`) for affirmative approval transitions from the Engineer lane ($x=3.1$) across to the Digital Twin lane ($x=8.25$):

```latex
\draw[flow] (approve.east) -| node[pos=0.35, above, font=\footnotesize]{Yes} (apply.north);
```

---

## 4. Verification & Outcome

- The resulting activity diagram compiled cleanly in LaTeX with zero clipping or text collisions.
- The diagram clearly reflects the **Human-in-the-Loop** safety mandate, demonstrating that an action can never reach the simulation or work-order stage without explicit authorization.
