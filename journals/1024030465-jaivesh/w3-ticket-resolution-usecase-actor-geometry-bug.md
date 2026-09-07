# Ticket 03: TikZ Actor Stick Figure Geometry & Elongated Coordinate Anomaly

- **Owner:** Jaivesh (1024030465)
- **Module:** Section 2 -- Use-Case Diagrams
- **Milestone:** Sprint 1 / Visual Modeling Verification

---

## 1. Problem Description & Difficulty

During document compilation, the primary actor in the Use-Case Diagram (**Maintenance Engineer**) exhibited an anomaly where the stick-figure legs extended downward across nearly the entire height of the system boundary rectangle, crossing over five use-case ellipses and overlapping the actor's text label.

### Symptoms:
- The PDF compiled with exit code 0 (no LaTeX syntax error).
- Visually, the left actor appeared with distorted "long legs" extending through the diagram.
- Text label `Maintenance Engineer` was drawn directly over the middle of the leg lines.
- Right actor (`Digital Twin System Actor`) was proportioned correctly.

---

## 2. Technical Context & Root Cause Analysis

Investigating the TikZ coordinate definitions for the left actor revealed a typographical error in the leg endpoints:

```latex
% Left Actor: Maintenance Engineer (BUGGY CODE)
\node (eng_head) at (-6.0, 1.4) {};
\draw[thick] (eng_head) circle (0.3);
\draw[thick] (-6.0, 1.1) -- (-6.0, 0.2);              % Torso
\draw[thick] (-6.55, 0.85) -- (-5.45, 0.85);          % Arms
\draw[thick] (-6.0, 0.2) -- (-6.35, -5.55);          % <-- BUG: -5.55 instead of -0.55
\draw[thick] (-6.0, 0.2) -- (-5.65, -5.55);          % <-- BUG: -5.55 instead of -0.55
\node[font=\small] at (-6.0, -1.05) {Maintenance\\Engineer};
```

Because `-5.55` was typed instead of `-0.55`, TikZ drew the legs with a vertical length of $5.75\text{ units}$ (10 times the intended height), extending all the way past use case 6 and 7.

---

## 3. Resolution & Engineering Implementation

I revised the endpoint coordinates of both left and right legs to match the geometry of the right actor:

```latex
% Left Actor: Maintenance Engineer (FIXED)
\node (eng_head) at (-6.0, 1.4) {};
\draw[thick] (eng_head) circle (0.3);
\draw[thick] (-6.0, 1.1) -- (-6.0, 0.2);              % Torso: height 0.9
\draw[thick] (-6.55, 0.85) -- (-5.45, 0.85);          % Arms: span 1.1
\draw[thick] (-6.0, 0.2) -- (-6.35, -0.55);          % Left leg: endpoint y = -0.55
\draw[thick] (-6.0, 0.2) -- (-5.65, -0.55);          % Right leg: endpoint y = -0.55
\node[font=\small] at (-6.0, -1.05) {Maintenance\\Engineer};
\coordinate (eng_hand) at (-5.45, 0.85);
```

---

## 4. Verification & Outcome

- Restored perfect bilateral symmetry between the human actor and system actor.
- All association connectors from `(eng_hand)` to use-case ellipses (`Upload Data`, `View Health`, `Recommend Action`, `Approve Action`, `Generate Alert`) now originate cleanly from hand height ($y=0.85$).
- Confirmed on Overleaf recompile; zero visual artifacts.
