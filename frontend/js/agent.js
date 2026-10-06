/**
 * agent.js — AI Agent and Approval UI Module
 * Autonomous Agentic Digital Twin for Turbofan Engines
 */

'use strict';

// Shared state for selected intervention
let selectedInterventionId = null;

/**
 * updateAgentPanel(analysis) — updates the agent diagnosis, indicators, explanation, and reasoning.
 * @param {Object} analysis - Output from /api/agent/analyze
 */
function updateAgentPanel(analysis) {
  if (!analysis) return;

  // 1. Identified Problem
  const problemEl = document.getElementById('agentProblem');
  if (problemEl) {
    const problemText = analysis.identified_problem || 'No degradation problems detected.';
    const isCritical = (analysis.identified_problem || '').toUpperCase().includes('CRITICAL') ||
                       (analysis.identified_problem || '').toUpperCase().includes('FAILURE');
    const badgeClass = isCritical ? 'text-red' : 'text-blue';
    problemEl.innerHTML = `
      <div class="problem-content ${badgeClass}">
        <i class="fa-solid ${isCritical ? 'fa-triangle-exclamation' : 'fa-circle-info'}"></i>
        <span>${escapeHtml(problemText)}</span>
      </div>
    `;
  }

  // 2. Degradation Indicators
  renderDegradationIndicators(analysis.degradation_indicators || []);

  // 3. AI Explanation
  const explanationEl = document.getElementById('agentExplanation');
  if (explanationEl) {
    const explanationText = analysis.explanation || 'Normal operating envelope maintained.';
    explanationEl.innerHTML = `
      <p class="explanation-text">${escapeHtml(explanationText)}</p>
    `;
  }

  // 4. Reasoning Steps
  renderReasoningSteps(analysis.reasoning_steps || []);

  // 5. If recommended intervention provided, set it as default selection
  if (analysis.recommended_intervention_id) {
    selectedInterventionId = analysis.recommended_intervention_id;
  }
}

/**
 * renderDegradationIndicators(indicators) — renders list of engine indicators with severity styling.
 * @param {Array<string>} indicators
 */
function renderDegradationIndicators(indicators) {
  const container = document.getElementById('degradationIndicators');
  if (!container) return;

  if (!indicators || indicators.length === 0) {
    container.innerHTML = '<li class="normal"><i class="fa-solid fa-check text-green"></i> All monitored parameters within nominal limits.</li>';
    return;
  }

  container.innerHTML = indicators.map(ind => {
    const isAbnormal = ind.toLowerCase().includes('abnormal') ||
                       ind.toLowerCase().includes('high') ||
                       ind.toLowerCase().includes('elevated') ||
                       ind.toLowerCase().includes('accelerat') ||
                       ind.toLowerCase().includes('exceed') ||
                       ind.toLowerCase().includes('critical') ||
                       ind.toLowerCase().includes('warning');
    const icon = isAbnormal ? 'fa-triangle-exclamation text-yellow' : 'fa-check text-green';
    const cls = isAbnormal ? 'abnormal' : 'normal';
    return `<li class="${cls}"><i class="fa-solid ${icon}"></i> <span>${escapeHtml(ind)}</span></li>`;
  }).join('');
}

/**
 * renderReasoningSteps(steps) — renders step-by-step reasoning list.
 * @param {Array<string>} steps
 */
function renderReasoningSteps(steps) {
  const container = document.getElementById('reasoningSteps');
  if (!container) return;

  if (!steps || steps.length === 0) {
    container.innerHTML = '<li class="placeholder-text">Reasoning steps will appear here after analysis.</li>';
    return;
  }

  container.innerHTML = steps.map((step, idx) => `
    <li>
      <strong>Step ${idx + 1}:</strong> ${escapeHtml(step.replace(/^Step\s*\d+:\s*/i, ''))}
    </li>
  `).join('');
}

/**
 * renderApprovalPanel(analysis, twinState) — renders the current recommendation and approval controls.
 * @param {Object|null} analysis
 * @param {Object|null} twinState
 */
function renderApprovalPanel(analysis, twinState) {
  const recNameEl = document.getElementById('recName');
  const recExplEl = document.getElementById('recExplanation');
  const recScoreEl = document.getElementById('recScore');
  const btnApprove = document.getElementById('btnApprove');
  const btnReject = document.getElementById('btnReject');

  if (!analysis) {
    if (recNameEl) recNameEl.textContent = 'No recommendation yet — run AI analysis first.';
    if (recExplEl) recExplEl.textContent = '';
    if (recScoreEl) recScoreEl.textContent = '—';
    if (btnApprove) btnApprove.disabled = true;
    if (btnReject) btnReject.disabled = true;
    return;
  }

  // Find the recommended or currently selected intervention
  const targetId = selectedInterventionId || analysis.recommended_intervention_id;
  const interventions = analysis.interventions || [];
  const selectedIntervention = interventions.find(i => i.intervention_id === targetId) ||
                              interventions.find(i => i.is_recommended) ||
                              interventions[0];

  if (selectedIntervention) {
    selectedInterventionId = selectedIntervention.intervention_id;
    if (recNameEl) {
      recNameEl.innerHTML = `
        <span class="rec-title">${escapeHtml(selectedIntervention.intervention_name)}</span>
        <span class="badge ${selectedIntervention.risk === 'Low' ? 'badge-green' : selectedIntervention.risk === 'Medium' ? 'badge-yellow' : 'badge-red'}">
          Risk: ${escapeHtml(selectedIntervention.risk)}
        </span>
      `;
    }

    if (recExplEl) {
      const gainText = selectedIntervention.rul_gain >= 0 ? `+${Math.round(selectedIntervention.rul_gain)} cycles` : `${Math.round(selectedIntervention.rul_gain)} cycles`;
      recExplEl.innerHTML = `
        <p><strong>Predicted Effect:</strong> RUL ${Math.round(selectedIntervention.current_rul)} → <strong>${Math.round(selectedIntervention.predicted_rul_after)} cycles</strong> (${gainText}, ${selectedIntervention.improvement_pct.toFixed(1)}% improvement).</p>
        <p class="text-secondary" style="font-size:0.8rem; margin-top:0.3rem;"><em>Cost: ${escapeHtml(selectedIntervention.cost)} | Source: ${escapeHtml(selectedIntervention.source)}</em></p>
      `;
    }

    if (recScoreEl) {
      const score = selectedIntervention.recommendation_score != null ? selectedIntervention.recommendation_score.toFixed(3) : '—';
      recScoreEl.textContent = score;
    }

    if (btnApprove) btnApprove.disabled = false;
    if (btnReject) btnReject.disabled = false;
  }

  // Check if twin already has a recorded decision status
  if (twinState && twinState.pending_recommendation) {
    const status = twinState.pending_recommendation.approval_status;
    if (status && typeof updateApprovalStatusBadge === 'function') {
      updateApprovalStatusBadge(status);
    }
  }
}

/**
 * renderHistoryTable(history) — populates the intervention history table.
 * @param {Array<Object>} history
 */
function renderHistoryTable(history) {
  const tbody = document.getElementById('historyTableBody');
  const countBadge = document.getElementById('historyCount');
  if (!tbody) return;

  if (countBadge) {
    countBadge.textContent = `${history.length} records`;
  }

  if (!history || history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No intervention history yet.</td></tr>';
    return;
  }

  tbody.innerHTML = history.slice().reverse().map(item => {
    const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : '—';
    const statusCls = (item.status || 'applied').toLowerCase() === 'approved' || (item.status || '').toLowerCase() === 'applied' ? 'text-green' : 'text-red';
    const rulBefore = item.rul_before != null ? Math.round(item.rul_before) : '—';
    const rulAfter = item.rul_after_predicted != null ? Math.round(item.rul_after_predicted) : '—';

    return `
      <tr>
        <td>${escapeHtml(timeStr)}</td>
        <td>${item.cycle != null ? item.cycle : '—'}</td>
        <td><strong>${escapeHtml(item.intervention_name || item.intervention_id || '—')}</strong></td>
        <td>${rulBefore}</td>
        <td><strong>${rulAfter}</strong></td>
        <td><span class="${statusCls}">${escapeHtml(item.status || 'Applied')}</span></td>
        <td class="history-notes">${escapeHtml(item.engineer_notes || '—')}</td>
      </tr>
    `;
  }).join('');
}

/**
 * Highlight a specific recommended row in the intervention table.
 * @param {string} interventionId
 */
function highlightRecommendedRow(interventionId) {
  const rows = document.querySelectorAll('#interventionTable tbody tr');
  rows.forEach(r => {
    if (r.dataset.interventionId === interventionId) {
      r.classList.add('selected');
    } else {
      r.classList.remove('selected');
    }
  });
}
