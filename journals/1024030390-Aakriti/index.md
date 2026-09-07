# Engineering Journal: Cyber-Physical Scoping, Dual-Actor Use Cases, DFDs & IEEE SRS

- **Student Name:** Aakriti
- **Roll Number:** 1024030390
- **Group:** 3C25 (BE CoE, 3rd Year)
- **Course:** UCS503 - Software Engineering Lab
- **Assigned Project Scope:** **Section 1 (Project Selection Phase & System Overview) and Section 2 (Analysis Phase: Dual-Actor Use Cases, 3-Lane Swimlane Control, DFD Process 3.0 Decomposition, IEEE SRS & Story Card US-03)**
  - **Direct Report References:**
    - **Chapter 1, Section 1.1:** Software Bid, Problem Domain, 3-Tier Feasibility Analysis & Operational Risk Management
    - **Chapter 1, Section 1.2:** Project Overview, Scope Demarcation (21-Channel Sensors, 30-Cycle Buffering), Out-of-Scope Boundaries & Multi-Stakeholder Analysis
    - **Chapter 2, Section 2.1:** Dual-Actor Use Case Modeling, Figure 2.1 (Include/Extend Semantics), Detailed Use Case Specifications (UC-01 & UC-02)
    - **Chapter 2, Section 2.2:** Multi-Lane Activity & Swimlane Diagram (Figure 2.2: Closed-Loop Agentic Control across Engineer, Twin, and ERP/ECU)
    - **Chapter 2, Section 2.3:** Hierarchical Data Flow Architecture: DFD Level 0 Context, DFD Level 1, and Level 2 Functional Decomposition of Process 3.0 (Agent Policy & Derate)
    - **Chapter 2, Section 2.4:** IEEE-830 Software Requirement Specification (FR-1 through FR-7, NFR-1 through NFR-4)
    - **Chapter 2, Section 2.5:** Agile User Stories Matrix (US-01 through US-05) & Formal Story Card US-03 (Acceptance Criteria 1–3)

---

## Weekly Ticket & Milestone Index

| Ticket / Week | Focus Area & Report Reference | Core Technical Challenge | Status |
| :--- | :--- | :--- | :--- |
| [**Ticket 01**](./w1-ticket-resolution-scoping-and-feasibility.md) | **Cyber-Physical Scoping & Feasibility**<br>*(Ref: Report Sec 1.1 & 1.2)* | Mitigating dataset dimensionality explosion across C-MAPSS sub-datasets, formalizing 3-tier feasibility, and defining the cyber-physical boundary. | **Resolved** |
| [**Ticket 02**](./w2-ticket-resolution-dual-actor-usecases.md) | **Dual-Actor Use Cases (UC-01 & UC-02)**<br>*(Ref: Report Sec 2.1, Fig 2.1, UC-01, UC-02)* | Modeling the interaction boundary between human engineers and autonomous background policy agents with strict `<<include>>` and `<<extend>>` UML semantics. | **Resolved** |
| [**Ticket 03**](./w3-ticket-resolution-closed-loop-swimlane.md) | **Closed-Loop 3-Lane Swimlane Workflow**<br>*(Ref: Report Sec 2.2, Fig 2.2)* | Modeling asynchronous decision flows across Maintenance Engineer, Digital Twin, and Enterprise ERP/ECU domains without race conditions or infinite loops. | **Resolved** |
| [**Ticket 04**](./w4-ticket-resolution-dfd-process3-decomposition.md) | **DFD Process 3.0 Hierarchical Decomposition**<br>*(Ref: Report Sec 2.3, Fig 2.3, 2.4, 2.5)* | Decomposing the cognitive AI agent policy into deterministic data flows (Processes 3.1, 3.2, 3.3) and preserving data store conservation across levels. | **Resolved** |
| [**Ticket 05**](./w5-ticket-resolution-ieee-srs-and-story-card-us03.md) | **IEEE SRS & Agile Story Card US-03**<br>*(Ref: Report Sec 2.4 & 2.5)* | Codifying non-linear degradation physics ($\\text{Derate}^{2.5}$) into verifiable IEEE requirements (FR-1 to FR-7) and formulating Story Card US-03 acceptance criteria. | **Resolved** |

---

## Scope & Contribution Summary
My primary responsibilities for the Mid-Semester Prototype Stage in Group 3C25 spanned the core foundations of **Project Selection, System Scoping, Dual-Actor Behavioral Modeling, Data Flow Architecture, and Requirements Engineering**:
1. **Strategic Scoping & Feasibility Modeling (Report Chapter 1):** Authored the Problem Domain definition, established the 3-tier feasibility framework (Technical, Economic, Time), and established the boundary constraints (21 sensor channels, 30-cycle sliding sequences on FD001) while cleanly separating out-of-scope physical hardware concerns.
2. **Dual-Actor Governance & Use Case Specifications (Report Section 2.1):** Established the architectural separation between the human **Maintenance Engineer** and the background **Autonomous Policy Agent**, designing Figure 2.1 and authoring complete specifications for **UC-01 (Predict RUL & Engine State)** and **UC-02 (Execute Closed-Loop Thrust Derating)**.
3. **Closed-Loop Control Flow Engineering (Report Section 2.2):** Formulated the 3-lane activity swimlane (Maintenance Engineer, Autonomous Agentic Twin, Enterprise ERP / ECU) orchestrating the complete lifecycle from sensor streaming to conditional 10% thrust derating and ERP dispatch.
4. **Hierarchical Data Flow Architecture (Report Section 2.3):** Designed DFD Level 0 Context, DFD Level 1 System decomposition, and authored the granular Level 2 functional decomposition of **Process 3.0 (Agent Policy & Derate)** into sub-processes 3.1, 3.2, and 3.3 with data stores D1 (SensorBuffer) and D2 (RULTelemetry).
5. **IEEE-830 Requirements & Agile Story Modeling (Report Sections 2.4 & 2.5):** Codified 7 testable Functional Requirements (incorporating multi-tier risk thresholds and empirical wear-reduction formulas) and 4 Non-Functional Requirements, formulated User Stories US-01 through US-05, and authored the formal Story Card for **US-03**.
