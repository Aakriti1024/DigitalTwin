/**
 * simulation.js — Virtual What-If Simulation and Candidate Intervention Module
 * Autonomous Agentic Digital Twin for Turbofan Engines
 */

'use strict';

let comparisonChart = null;
let currentSimData = null;

/**
 * initSimulation() — initializes the what-if simulation panel and chart.
 */
function initSimulation() {
  createComparisonChart();
}

/**
 * createComparisonChart() — builds the Chart.js line chart for baseline vs intervention comparison.
 */
function createComparisonChart() {
  const canvas = document.getElementById('comparisonChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  if (comparisonChart) {
    comparisonChart.destroy();
    comparisonChart = null;
  }

  comparisonChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Baseline (No Action)',
          data: [],
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.08)',
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
          tension: 0.3,
        },
        {
          label: 'After Intervention',
          data: [],
          borderColor: '#22c55e',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display: false, // Custom HTML legend already in index.html
        },
        tooltip: {
          backgroundColor: '#1e293b',
          borderColor: '#334155',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          callbacks: {
            title: function (items) {
              return `Cycle +${items[0].label}`;
            },
            label: function (ctx) {
              return `${ctx.dataset.label}: ${Math.round(ctx.parsed.y)} cycles RUL`;
            },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Future Simulation Cycles',
            color: '#94a3b8',
            font: { size: 11 },
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
          },
          ticks: {
            color: '#64748b',
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: 'Predicted RUL (cycles)',
            color: '#94a3b8',
            font: { size: 11 },
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.05)',
          },
          ticks: {
            color: '#64748b',
          },
          min: 0,
        },
      },
    },
  });
}

/**
 * populateInterventionSelect(interventions) — populates the what-if dropdown.
 * @param {Array<Object>} interventions
 */
function populateInterventionSelect(interventions) {
  const select = document.getElementById('whatIfSelect');
  if (!select) return;

  select.innerHTML = '<option value="">— Select intervention —</option>';

  if (!interventions || interventions.length === 0) return;

  interventions.forEach(item => {
    const opt = document.createElement('option');
    opt.value = item.intervention_id;
    const gainStr = item.rul_gain >= 0 ? `+${Math.round(item.rul_gain)}` : `${Math.round(item.rul_gain)}`;
    opt.textContent = `${item.intervention_name} (${gainStr} cycles RUL)`;
    if (item.is_recommended) {
      opt.selected = true;
    }
    select.appendChild(opt);
  }

  );
  select.onchange = () => {
    selectedInterventionId = select.value || null;

    console.log("DROPDOWN VALUE:", select.value);
    console.log("SELECTED INTERVENTION:", selectedInterventionId);
  };
}

/**
 * runWhatIfSimulation() — calls the what-if simulation API for the selected intervention.
 */
async function runWhatIfSimulation() {
  const select = document.getElementById('whatIfSelect');
  const btn = document.getElementById('btnWhatIf');
  if (!select) return;

  const interventionId = select.value;
  if (!interventionId) {
    if (typeof showToast === 'function') {
      showToast('Please select an intervention to simulate.', 'warning');
    }
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Simulating…';
  }

  try {
    const url = `${API_BASE}/simulation/what-if`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine_id: currentEngineId,
        intervention_id: interventionId,
        n_future_cycles: 50,
      }),
    });

    if (!response.ok) {
      throw new Error(`Simulation request failed: ${response.status}`);
    }

    const data = await response.json();
    currentSimData = data;

    updateComparisonChart(data);
    updateSimSummaryCards(data);

    if (typeof showToast === 'function') {
      showToast(`Simulation complete: ${data.intervention_name}`, 'success');
    }
  } catch (err) {
    console.error('[What-If Simulation] Error:', err);
    if (typeof showToast === 'function') {
      showToast(`Simulation failed: ${err.message}`, 'error');
    }
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Simulation';
    }
  }
}

/**
 * updateComparisonChart(simResult) — updates the Chart.js baseline vs intervention lines.
 * @param {Object} simResult
 */
function updateComparisonChart(simResult) {
  if (!comparisonChart) {
    createComparisonChart();
  }
  if (!comparisonChart || !simResult) return;

  const baseline = simResult.rul_trajectory_baseline || [];
  const intervention = simResult.rul_trajectory_intervention || [];
  const n = Math.max(baseline.length, intervention.length);

  const labels = Array.from({ length: n }, (_, i) => i);

  comparisonChart.data.labels = labels;
  comparisonChart.data.datasets[0].data = baseline;
  comparisonChart.data.datasets[1].data = intervention;
  comparisonChart.update();
}

/**
 * updateSimSummaryCards(simResult) — fills in the summary numbers.
 * @param {Object} simResult
 */
function updateSimSummaryCards(simResult) {
  if (!simResult) return;

  const curEl = document.getElementById('simCurrentRUL');
  const predEl = document.getElementById('simPredictedRUL');
  const gainEl = document.getElementById('simRULGain');
  const impEl = document.getElementById('simImprovement');

  if (curEl) curEl.textContent = Math.round(simResult.current_rul);
  if (predEl) predEl.textContent = Math.round(simResult.predicted_rul_after);

  if (gainEl) {
    const gain = Math.round(simResult.rul_gain);
    gainEl.textContent = gain >= 0 ? `+${gain}` : `${gain}`;
    gainEl.style.color = gain >= 0 ? '#22c55e' : '#ef4444';
  }

  if (impEl) {
    const imp = simResult.improvement_pct != null ? simResult.improvement_pct.toFixed(1) : '0.0';
    impEl.textContent = `${imp}%`;
  }
}

/**
 * renderInterventionTable(interventions) — renders the table of candidate interventions.
 * @param {Array<Object>} interventions
 */
function renderInterventionTable(interventions) {
  const tbody = document.getElementById('interventionTableBody');
  if (!tbody) return;

  if (!interventions || interventions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-row">Run analysis to populate intervention candidates.</td></tr>';
    return;
  }

  tbody.innerHTML = interventions.map((item, index) => {
    const isRec = item.is_recommended;
    const isHighRisk = item.risk === 'High' || item.risk === 'Critical';
    const rowClass = isRec ? 'row-recommended' : isHighRisk ? 'row-high-risk' : 'row-possible';
    const gainFormatted = item.rul_gain >= 0 ? `+${Math.round(item.rul_gain)}` : `${Math.round(item.rul_gain)}`;
    const scoreFormatted = item.recommendation_score != null ? item.recommendation_score.toFixed(3) : '—';
    const riskBadgeClass = item.risk === 'Low' ? 'badge-green' : item.risk === 'Medium' ? 'badge-yellow' : 'badge-red';

    return `
      <tr class="${rowClass}" data-intervention-id="${escapeHtml(item.intervention_id)}" onclick="selectInterventionFromTable('${escapeHtml(item.intervention_id)}')">
        <td><strong>#${item.rank || index + 1}</strong></td>
        <td>
          <strong>${escapeHtml(item.intervention_name)}</strong>
          ${isRec ? '<span class="rec-star"><i class="fa-solid fa-star"></i> AI Pick</span>' : ''}
        </td>
        <td>${Math.round(item.predicted_rul_after)} cycles</td>
        <td><strong class="${item.rul_gain >= 0 ? 'text-green' : 'text-red'}">${gainFormatted}</strong></td>
        <td><span class="badge ${riskBadgeClass}">${escapeHtml(item.risk)}</span></td>
        <td>${escapeHtml(item.cost)}</td>
        <td>${scoreFormatted}</td>
        <td>${isRec ? '<i class="fa-solid fa-check text-green"></i> <strong>Best</strong>' : '—'}</td>
      </tr>
    `;
  }).join('');
}

/**
 * selectInterventionFromTable(interventionId) — row click handler to sync selection across panels.
 * @param {string} interventionId
 */
function selectInterventionFromTable(interventionId) {
  selectedInterventionId = interventionId;

  // Sync what-if select
  const select = document.getElementById('whatIfSelect');
  if (select) {
    select.value = interventionId;
  }

  // Highlight table row
  if (typeof highlightRecommendedRow === 'function') {
    highlightRecommendedRow(interventionId);
  }

  // Update approval panel
  if (typeof updateApprovalPanel === 'function' && typeof currentAnalysis !== 'undefined') {
    updateApprovalPanel(currentAnalysis, typeof currentTwinState !== 'undefined' ? currentTwinState : null);
  }

  // Auto-run what-if simulation for the clicked row
  runWhatIfSimulation();
}
