# Ticket 02: TikZ Vector Architecture & Sprint Scheduling for Agile Scrum SDLC

- **Owner:** Anoushka (1024030267)
- **Module:** Section 1 -- Software Development Life Cycle (SDLC) Modeling & Sprint Planning
- **Report Reference:**
  - **Chapter 1, Section 1.3 (Figure 1.1):** *Iterative Agile SDLC Workflow for Cyber-Physical Digital Twin (TikZ vector diagram).*
  - **Chapter 1, Section 1.3 (Sprint Breakdown):** *Sprint 1 (Weeks 1–3), Sprint 2 (Weeks 4–6), Sprint 3 (Weeks 7–9), and Sprint 4 (Weeks 10–12).*
- **Milestone:** Sprint 1 / Process Modeling & Methodology Formalization

---

## 1. Problem Description & Difficulty

In academic software engineering deliverables, standard practice often defaults to pasting low-resolution raster screenshots of generic Agile workflows found online. Such images degrade visual quality, do not reflect project-specific milestones, and are heavily penalized during evaluation.

For our project report, I was responsible for designing and implementing a custom, publication-quality **TikZ vector diagram (Figure 1.1)** in LaTeX representing our **Iterative & Agile Scrum SDLC Model**, as well as scheduling the 12-week development lifecycle into four synchronized sprints.

### Technical Challenges in TikZ:
1. **Geometric Arrow Collisions:** The workflow consists of four perimeter sprint boxes arranged cyclically, with a central dashed node representing **Agile Scrum Feedback**. Initial curved arrow paths (`to[out=..., in=...]`) pierced directly through the central feedback circle, creating unreadable line collisions.
2. **Page Width & Margin Constraints:** Standard A4 margins ($1\text{ in} = 2.54\text{ cm}$) restrict text width to $6.27\text{ in} \approx 15.9\text{ cm}$. Sizing four sprint nodes (each containing bold titles and two-line subtitles) caused horizontal margin overflow (`Overfull \hbox by 46.2pt`).
3. **Color Contrast & Readability:** Ensuring that distinct sprint fills (`blue!10`, `green!10`, `orange!10`, `purple!10`) retained clear contrast in both color PDF viewers and black-and-white printouts.

---

## 2. Technical Context & Root Cause Analysis

The initial TikZ layout attempted relative node positioning and unconstrained curved paths:

```latex
% DEFICIENT INITIAL TIKZ CODE: Relative positioning causing arrow overlaps
\begin{tikzpicture}
    \node[draw] (s1) {Sprint 1: Data Ingestion};
    \node[draw, right=of s1] (s2) {Sprint 2: PyTorch LSTM};
    \node[draw, below=of s2] (s3) {Sprint 3: Agent Policy};
    \node[draw, left=of s3] (s4) {Sprint 4: UI Integration};
    \node[draw, circle] (fb) at (2, -1) {Feedback};
    
    % Curved arrows collided with the central node 'fb'
    \draw[->] (s1) to[bend left=20] (s2);
    \draw[->] (s2) to[bend left=20] (s3);
    \draw[->] (s3) to[bend left=20] (s4);
    \draw[->] (s4) to[bend left=20] (s1);
\end{tikzpicture}
```

Because node dimensions expanded automatically with subtitle length, the right boundary extended past $17.5\text{ cm}$, pushing `s2` and `s3` beyond the right page margin. Furthermore, the curved paths intersected the central circle at unpredictable coordinates.

---

## 3. Resolution & Engineering Implementation

### A. Orthogonal 2x2 Perimeter Grid with Scaling Transform
I reconstructed the diagram using an explicit orthogonal coordinate matrix and a unified scaling factor (`scale=0.92, transform shape`). I defined fixed node dimensions (`minimum width=4.8cm, minimum height=1.2cm`) with uniform rounded corners (`rounded corners=6pt`):

```latex
% OPTIMIZED TIKZ ARCHITECTURE (Report Figure 1.1)
\begin{figure}[H]
\centering
\begin{tikzpicture}[
    scale=0.92, transform shape,
    every node/.style={font=\small},
    sprint/.style={draw, rounded corners=6pt, align=center, minimum width=4.8cm, minimum height=1.2cm, thick}
]
    % 4-Corner Rectangular Topology
    \node[sprint, fill=blue!10]   (s1) at (0, 2.6)   {\textbf{Sprint 1: Data \& Ingestion}\\ \footnotesize Cleaning, Scaling \& Sliding Windows};
    \node[sprint, fill=green!10]  (s2) at (6.8, 2.6) {\textbf{Sprint 2: PyTorch LSTM Twin}\\ \footnotesize Model Training \& RUL Estimation};
    \node[sprint, fill=orange!10] (s3) at (6.8, 0)   {\textbf{Sprint 3: Agent Policy Engine}\\ \footnotesize Prescriptive Actions \& Closed Loop};
    \node[sprint, fill=purple!10] (s4) at (0, 0)     {\textbf{Sprint 4: UI \& Integration}\\ \footnotesize Streamlit Control Center \& Dispatch};

    % Clean Perimeter Vector Paths with Dedicated Sprint Colors
    \draw[-Latex, very thick, blue!80!black]   (s1.east)  -- (s2.west);
    \draw[-Latex, very thick, green!70!black]  (s2.south) -- (s3.north);
    \draw[-Latex, very thick, orange!80!black] (s3.west)  -- (s4.east);
    \draw[-Latex, very thick, purple!80!black] (s4.north) -- (s1.south);

    % Centrally Centered Feedback Circle (Zero Collisions)
    \node[draw, dashed, circle, fill=gray!5, minimum size=2.0cm, align=center, font=\scriptsize] 
          at (3.4, 1.3) {\textbf{Agile}\\ \textbf{Scrum}\\ \textbf{Feedback}};
\end{tikzpicture}
\caption{Iterative Agile SDLC Workflow for Cyber-Physical Digital Twin}
\label{fig:sdlc}
\end{figure}
```

### B. 12-Week Milestone Scheduling (Report Section 1.3)
In parallel with the visual model, I formulated the detailed 4-sprint breakdown establishing milestone deliverables for each team member:
- **Sprint 1 (Weeks 1–3): Data & Ingestion** – C-MAPSS data exploration, constant sensor filtering, feature normalization, and sliding-window matrix generation.
- **Sprint 2 (Weeks 4–6): PyTorch LSTM Twin** – 2-layer LSTM regression network design, hyperparameter tuning (hidden size, learning rate), loss curve validation, and RMSE scoring.
- **Sprint 3 (Weeks 7–9): Agent Policy Engine** – Risk-cost matrix, health classification thresholds (Caution, Warning, Critical), and closed-loop degradation physics modeling.
- **Sprint 4 (Weeks 10–12): UI & Integration** – Streamlit control center wireframes, parameter injection sliders, Human-in-the-Loop review gates, and JSON work-order dispatch formatting.

---

## 4. Verification & Outcome

1. **Zero LaTeX Compiler Warnings:** The revised TikZ code compiled with exit code 0 and zero `Overfull \hbox` warnings.
2. **Geometric Harmony:** The total canvas width ($11.6\text{ cm}$) fits centered inside standard A4 page margins with $> 2.1\text{ cm}$ clearance on each side.
3. **Publication Quality:** The diagram renders with razor-sharp vector typography in PDF viewers at any zoom level, clearly conveying our software development methodology during project evaluation.
