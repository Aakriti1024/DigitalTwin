# Ticket 05: Agile User Story Mapping, Story Point Sizing & Story Card US-05 Formulation

- **Owner:** Anoushka (1024030267)
- **Module:** Section 2 -- User Stories, Story Points & Agile Story Cards
- **Report Reference:**
  - **Chapter 2, Section 2.4 (User Stories Matrix):** *Formalization of User Stories US-01 through US-06 with Priority and Fibonacci Story Points.*
  - **Chapter 2, Section 2.4 (Sample Story Card -- US-05):** *Story specification and Acceptance Criteria (1 through 5) for closed-loop Time-on-Wing evaluation.*
- **Milestone:** Sprint 2 / Agile Backlog Grooming & Acceptance Engineering

---

## 1. Problem Description & Difficulty

In an Agile Scrum development lifecycle, functional requirements must be translated into **User Stories** that capture user personas, functional objectives, and business value. Furthermore, to guide sprint allocation, stories must be assigned realistic effort estimates using **Fibonacci Story Points** ($1, 2, 3, 5, 8, 13$).

For our project report, I was responsible for:
1. Formulating the core **User Stories Matrix (US-01 through US-06)** across two primary user personas: the **Maintenance Engineer** and the **Data Science Team**.
2. Authoring the detailed **Sample Story Card for US-05**, defining rigorous, testable **Acceptance Criteria** for the project’s core novelty: closed-loop degradation simulation and Time-on-Wing ($\Delta \text{TOW}$) evaluation.
3. Overcoming severe **LaTeX margin overflow and text-clipping errors** when rendering long multi-line story descriptions and acceptance criteria inside academic tables.

---

## 2. Technical Context & Root Cause Analysis

### A. LaTeX Table Margin Overflow (`Overfull \hbox`)
Standard LaTeX `tabular` environments with fixed column widths (`p{...}`) lack dynamic word-wrapping and hyphenation, causing long narrative sentences to extend past the right page margin:

```latex
% DEFICIENT TABLE INITIAL CODE: Margin overflow and rigid text clipping
\begin{table}[h]
\begin{tabular}{|c|p{8cm}|c|c|}
\hline
ID & User Story & Priority & Story Points \\
\hline
US-01 & As a Maintenance Engineer, I want to provide engine sensor data... & High & 3 \\
% Text clipped and extended 72pt into the right margin!
\hline
\end{tabular}
\end{table}
```

The LaTeX compiler issued repeated warnings: `Overfull \hbox (72.4pt too wide) in alignment at lines 531--542`.

### B. Vague Acceptance Criteria
Early drafts of **Story Card US-05** lacked verifiable acceptance steps:
- *"The simulation should show that the engine improves."*
- *"Engineer should see a graph of the result."*

These failed agile verification standards because they lacked step-by-step preconditions, actions, and measurable expected results.

---

## 3. Resolution & Engineering Implementation

### A. Custom Ragged-Right Column Architecture (`longtable` & `booktabs`)
To eliminate LaTeX margin overflow, I introduced a custom column specifier combining ragged-right alignment with automatic line-breaking, wrapping the table inside a `longtable` environment with publication-grade `booktabs` rules:

```latex
% In preamble:
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}

% In Section 2.4:
\begin{longtable}{L{2.5cm} L{4.5cm} L{2cm} L{2.7cm}}
\toprule
\textbf{ID} & \textbf{User Story} & \textbf{Priority} & \textbf{Story Points} \\
\midrule
US-01 & As a Maintenance Engineer, I want to provide engine sensor data so that the system can analyze the engine condition. & High & 3 \\
US-02 & As a Maintenance Engineer, I want to view the predicted RUL and health status of an engine so that I can plan maintenance. & High & 5 \\
US-03 & As a Maintenance Engineer, I want the system to generate a maintenance recommendation when the predicted RUL indicates increased risk so that I can take appropriate action. & High & 5 \\
US-04 & As a Maintenance Engineer, I want to review and approve or reject a recommended action so that maintenance decisions remain under human control. & High & 3 \\
US-05 & As a Maintenance Engineer, I want to see the effect of an approved action on the Digital Twin so that I can evaluate the resulting engine degradation and time-on-wing extension. & High & 5 \\
US-06 & As a Data Science Team member, I want to evaluate the RUL model using MAE, RMSE, and $R^2$ so that I can assess prediction performance. & Medium & 3 \\
\bottomrule
\end{longtable}
```

### B. Fibonacci Story Point Estimation Justification
I led the sizing sessions, allocating points based on technical complexity, architectural risk, and dependency depth:
- **US-01 (3 pts):** Standard CSV ingestion and validation; low algorithmic risk.
- **US-02 (5 pts):** Requires integrating sliding-window inference, feature scaling, and real-time dashboard presentation.
- **US-03 (5 pts):** Requires configuring multi-tier risk evaluation logic and explainable rationale generation.
- **US-04 (3 pts):** Implementation of the transactional approval/rejection decision gate.
- **US-05 (5 pts):** Highest algorithmic complexity; involves aero-thermal wear modeling, differential equation stepping, and comparative baseline graphing.
- **US-06 (3 pts):** Standard statistical evaluation pipeline over holdout test sets.
- **Total Sprint Velocity Budget:** $24\text{ Story Points}$, perfectly distributed across the team.

### C. Formal Acceptance Criteria for Story Card US-05
I authored the dedicated **Story Card for US-05**, formalizing the 5 sequential verification steps required for acceptance:

```latex
\subsection*{Sample Story Card -- US-05}

\begin{longtable}{L{3cm} L{9.5cm}}
\toprule
Story & As a Maintenance Engineer, I want to see the effect of an approved action 
        on the Digital Twin so that I can evaluate the resulting engine degradation 
        and time-on-wing extension. \\
Acceptance Criteria &
1. Given a recommended action, the engineer can approve the action.\\
& 2. The approved action is applied to the Digital Twin simulation.\\
& 3. The Digital Twin updates the simulated engine degradation trajectory.\\
& 4. The system compares the intervention trajectory with the passive baseline.\\
& 5. The system displays the estimated time-on-wing extension resulting from the approved action. \\
Priority & High \\
Estimate & 5 Story Points \\
\bottomrule
\end{longtable}
```

---

## 4. Verification & Outcome

1. **Flawless LaTeX Compilation:** All tables in Section 2.4 compile cleanly with zero margin overflow warnings, perfectly aligned borders, and crisp typography.
2. **Complete Agile Traceability:** Established seamless bi-directional links between Functional Requirements (`FR-7, FR-8, FR-9`) and `US-05`.
3. **Execution Blueprint for Sprint 4:** Story Card US-05 established the immutable criteria against which our upcoming closed-loop simulation and dashboard interface will be formally evaluated.
