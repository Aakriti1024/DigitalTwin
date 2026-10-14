/**
 * main.js — Digital Twin Dashboard
 * Main application entry point: polling, UI orchestration, button handlers.
 */

'use strict';

/* ------------------------------------------------------------------ */
/*  Global state                                                        */
const API_BASE = (typeof window !== 'undefined' && window.location && window.location.origin && window.location.origin !== 'null')
  ? `${window.location.origin}/api`
  : 'http://localhost:8000/api';
let pollingInterval = null;
let currentTwinState = null;
let currentAnalysis = null;
let currentSimResult = null;
let currentEngineId = 'E1';

/* ------------------------------------------------------------------ */
/*  Initialization                                                      */
/* ------------------------------------------------------------------ */

/**
 * init() — called on DOMContentLoaded.
 * Wires up the engine selector, polling, button listeners, and loads
 * the initial state.
 */
function init() {
  // --- Engine selector ---
  const engineSelect = document.getElementById('engineSelect');
  if (engineSelect) {
    engineSelect.value = currentEngineId;
    engineSelect.addEventListener('change', () => {
      currentEngineId = engineSelect.value;
      stopPolling();
      currentAnalysis = null;
      currentSimResult = null;
      // Reset agent + approval panels
      resetAgentPanel();
      startPolling();
    });
  }

  // --- Initialize charts and simulation modules ---
  if (typeof initCharts === 'function') initCharts();
  if (typeof initSimulation === 'function') initSimulation();

  // --- Button: Analyze ---
  const btnAnalyze = document.getElementById('btnAnalyze');
  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', handleAnalyze);
  }

  // --- Button: Approve ---
  const btnApprove = document.getElementById('btnApprove');
  if (btnApprove) {
    btnApprove.addEventListener('click', handleApprove);
  }

  // --- Button: Reject ---
  const btnReject = document.getElementById('btnReject');
  if (btnReject) {
    btnReject.addEventListener('click', handleReject);
  }

  // --- Button: What-If simulation ---
  const btnWhatIf = document.getElementById('btnWhatIf');
  if (btnWhatIf) {
    btnWhatIf.addEventListener('click', () => {
      if (typeof runWhatIfSimulation === 'function') runWhatIfSimulation();
    });
  }

  // --- Sensor toggle buttons ---
  document.querySelectorAll('.sensor-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sensor = btn.dataset.sensor;
      btn.classList.toggle('active');
      if (typeof toggleSensor === 'function') toggleSensor(sensor);
    });
  });

  // --- Replay control buttons ---
  const btnReplayPlay = document.getElementById('btnReplayPlay');
  if (btnReplayPlay) btnReplayPlay.addEventListener('click', handleReplayPlay);

  const btnReplayPause = document.getElementById('btnReplayPause');
  if (btnReplayPause) btnReplayPause.addEventListener('click', handleReplayPause);

  const btnReplayStep = document.getElementById('btnReplayStep');
  if (btnReplayStep) btnReplayStep.addEventListener('click', handleReplayStep);

  const btnReplayReset = document.getElementById('btnReplayReset');
  if (btnReplayReset) btnReplayReset.addEventListener('click', handleReplayReset);

  const replaySpeedSelect = document.getElementById('replaySpeedSelect');
  if (replaySpeedSelect) replaySpeedSelect.addEventListener('change', handleReplaySpeed);

  // --- Start polling and load first data ---
  startPolling();
  fetchEngineState();
}

/* ------------------------------------------------------------------ */
/*  Polling                                                             */
/* ------------------------------------------------------------------ */

/**
 * startPolling(intervalMs) — starts the auto-refresh loop.
 * @param {number} intervalMs  Polling cadence in milliseconds (default 3000).
 */
function startPolling(intervalMs = 3000) {
  stopPolling();
  pollingInterval = setInterval(fetchEngineState, intervalMs);
}

/**
 * stopPolling() — clears the current polling interval.
 */
function stopPolling() {
  if (pollingInterval !== null) {
    clearInterval(pollingInterval);
    pollingInterval = null;
  }
}

/* ------------------------------------------------------------------ */
/*  Data fetching                                                       */
/* ------------------------------------------------------------------ */

/**
 * fetchEngineState() — fetches current twin state from the API.
 * On success calls updateAllUI; on failure shows a connection error.
 */
async function fetchEngineState() {
  try {
    const url = `${API_BASE}/twin/replay/status?engine_id=${encodeURIComponent(currentEngineId)}`;
    const response = await fetch(url);

    if (response.ok) {
      const data = await response.json();
      currentTwinState = data.twin_state;
      updateConnectionStatus(true);
      updateAllUI(data.twin_state, data.status);
      return;
    }

    // Fallback to /twin/state if needed
    const fallbackUrl = `${API_BASE}/twin/state?engine_id=${encodeURIComponent(currentEngineId)}`;
    const fbResponse = await fetch(fallbackUrl);
    if (!fbResponse.ok) {
      throw new Error(`HTTP ${fbResponse.status}: ${fbResponse.statusText}`);
    }
    const fbData = await fbResponse.json();
    currentTwinState = fbData;
    updateConnectionStatus(true);
    updateAllUI(fbData);
  } catch (err) {
    console.error('[Digital Twin] fetchEngineState error:', err);
    updateConnectionStatus(false);
  }
}

/**
 * fetchSensorHistory() — fetches the rolling sensor/RUL history.
 * @param {number} nCycles  Number of historical cycles to request.
 * @returns {Promise<Array>}
 */

async function fetchSensorHistory(nCycles = 50) {
  const url = `${API_BASE}/twin/history?engine_id=${encodeURIComponent(currentEngineId)}&n_cycles=${nCycles}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const payload = await response.json();

  if (Array.isArray(payload)) {
    return payload;
  }

  const rulHistory = payload.rul_history || [];
  const sensorTrends = payload.sensor_trends || {};

  return rulHistory.map(point => {
    const sensors = {};

    for (const [sensorId, readings] of Object.entries(sensorTrends)) {
      const reading = readings.find(item => item.cycle === point.cycle);

      if (reading) {
        sensors[sensorId] = reading.value;
      }
    }

    return {
      ...point,
      sensors
    };
  });
}

/* ------------------------------------------------------------------ */
/*  UI orchestration                                                    */
/* ------------------------------------------------------------------ */

/**
 * updateAllUI(state, replayStatus) — central dispatch; updates every dashboard section.
 * @param {Object} state  Twin state object from the API.
 * @param {Object} [replayStatus] Replay status from the API.
 */
function updateAllUI(state, replayStatus = null) {
  updateReplayBar(state, replayStatus);
  updateTopStats(state);
  updateRULTrend(state);
  updateHealthGauge(state);
  updateSensorTrends();
  updateApprovalPanel(currentAnalysis, state);
  updateHistoryTable(state.intervention_history || []);
  updateMetricsTable(state.model_metrics || null);
  updateLastUpdated();
}

/**
 * updateReplayBar(state, replayStatus) — updates replay controls & indicators.
 */
function updateReplayBar(state, replayStatus) {
  if (!state) return;

  const statusBadge = document.getElementById('replayStatusBadge');
  const warmupBadge = document.getElementById('warmupBadge');
  const refRulVal = document.getElementById('refRulValue');
  const playBtn = document.getElementById('btnReplayPlay');

  if (refRulVal) {
    refRulVal.textContent = state.reference_rul != null
      ? `${Math.round(state.reference_rul)} cycles`
      : '—';
  }

  if (warmupBadge) {
    if (state.is_warmed_up) {
      warmupBadge.textContent = 'Window Full: Active Model Predictions (MC-Dropout)';
      warmupBadge.className = 'badge warmup-badge ready';
    } else {
      warmupBadge.textContent = `Collecting sensor history: ${state.warmup_progress || '0/30'} readings`;
      warmupBadge.className = 'badge warmup-badge';
    }
  }

  if (statusBadge) {
    const cycle = state.current_cycle || 0;
    const isPlaying = replayStatus ? replayStatus.is_playing : false;
    const total = replayStatus ? replayStatus.total_cycles : 31;
    const st = replayStatus ? replayStatus.status : (cycle > 0 ? 'paused' : 'ready');

    statusBadge.textContent = `${st.toUpperCase()} · Cycle ${cycle}/${total}`;

    if (playBtn) {
      if (isPlaying) {
        playBtn.classList.add('active-play');
        playBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Playing';
      } else {
        playBtn.classList.remove('active-play');
        playBtn.innerHTML = '<i class="fa-solid fa-play"></i> Play';
      }
    }
  }
}

/**
 * updateTopStats(state) — fills in the four top stat cards.
 * @param {Object} state
 */
function updateTopStats(state) {
  if (!state) return;

  // Health state badge
  const healthInfo = formatHealthState(state.health_state || 'unknown');
  const healthBadge = document.getElementById('healthStateBadge');
  if (healthBadge) {
    healthBadge.textContent = healthInfo.text;
    healthBadge.className = `health-badge ${healthInfo.className}`;
  }

  // Status message
  const statusMsg = document.getElementById('statusMessage');
  if (statusMsg) statusMsg.textContent = state.status_message || '';

  // Predicted RUL & Uncertainty
  const rulValue = document.getElementById('rulValue');
  const rulUnc = document.getElementById('rulUncertainty');

  if (rulValue) {
    if (!state.is_warmed_up) {
      rulValue.textContent = 'Warm-up';
      rulValue.style.fontSize = '1.7rem';
      rulValue.style.color = '#eab308';
      if (rulUnc) {
        rulUnc.textContent = `Collecting history: ${state.warmup_progress || '0/30'} readings`;
        rulUnc.style.color = '#eab308';
      }
    } else {
      const rul = state.predicted_rul != null ? Math.round(state.predicted_rul) : '—';
      rulValue.textContent = rul;
      rulValue.style.fontSize = '';
      rulValue.style.color = getRULColor(state.predicted_rul);

      if (rulUnc) {
        const lower = state.rul_lower != null ? Math.round(state.rul_lower) : null;
        const upper = state.rul_upper != null ? Math.round(state.rul_upper) : null;
        if (lower != null && upper != null) {
          rulUnc.textContent = `± ${Math.round((upper - lower) / 2)} cycles  (${lower}–${upper})`;
        } else {
          rulUnc.textContent = '± — cycles';
        }
        rulUnc.style.color = '';
      }
    }
  }

  // Risk badge
  const riskInfo = formatRisk(state.risk_level || 'unknown');
  const riskBadge = document.getElementById('riskBadge');
  if (riskBadge) {
    riskBadge.textContent = riskInfo.text;
    riskBadge.className = `risk-badge ${riskInfo.className}`;
  }
  const riskSub = document.getElementById('riskSubtext');
  if (riskSub) riskSub.textContent = riskInfo.subtext || '';

  // Health score + progress bar
  const score = state.health_score != null ? Math.round(state.health_score) : null;
  const healthScoreVal = document.getElementById('healthScoreValue');
  if (healthScoreVal) {
    healthScoreVal.textContent = score != null ? score : '—';
    healthScoreVal.style.color = score != null ? getHealthScoreColor(score) : '';
  }
  const progressBar = document.getElementById('healthProgressBar');
  if (progressBar) {
    progressBar.style.width = score != null ? `${score}%` : '0%';
    progressBar.style.background = score != null
      ? `linear-gradient(90deg, ${getHealthScoreGradient(score)})`
      : '';
  }
  const healthSub = document.getElementById('healthScoreSub');
  if (healthSub) healthSub.textContent = score != null
    ? `Cycle ${state.current_cycle || '—'} · Engine ${currentEngineId}`
    : '—';

  // Gauge center label
  const gaugeCenter = document.getElementById('gaugeCenterValue');
  if (gaugeCenter) gaugeCenter.textContent = score != null ? score : '—';
}

/* ------------------------------------------------------------------ */
/*  Chart update helpers (delegates to charts.js / simulation.js)      */
/* ------------------------------------------------------------------ */

/**
 * updateRULTrend(state) — fetches history and delegates to charts module.
 */
async function updateRULTrend(state) {
  try {
    const history = await fetchSensorHistory(50);
    if (typeof updateRULTrendChart === 'function') {
      updateRULTrendChart(history);
    }
  } catch (err) {
    console.warn('[Digital Twin] updateRULTrend:', err.message);
  }
}

/**
 * updateHealthGauge(state) — delegates to charts module.
 */
function updateHealthGauge(state) {
  if (!state) return;
  if (typeof updateHealthGaugeChart === 'function') {
    updateHealthGaugeChart(state.health_score || 0, state.health_state || 'unknown');
  }
}

/**
 * updateSensorTrends() — fetches history and delegates to charts module.
 */
async function updateSensorTrends() {
  try {
    const history = await fetchSensorHistory(50);
    if (typeof updateSensorTrendChart === 'function') {
      updateSensorTrendChart(history);
    }
  } catch (err) {
    console.warn('[Digital Twin] updateSensorTrends:', err.message);
  }
}

/* ------------------------------------------------------------------ */
/*  Intervention table & history                                        */
/* ------------------------------------------------------------------ */

/**
 * updateInterventionTable(analysis) — delegates rendering to agent.js.
 * @param {Object} analysis
 */
function updateInterventionTable(analysis) {
  if (!analysis) return;
  const interventions = analysis.interventions || analysis || [];
  if (typeof renderInterventionTable === 'function') {
    renderInterventionTable(Array.isArray(interventions) ? interventions : []);
  }
}

/**
 * updateHistoryTable(history) — delegates to agent.js.
 * @param {Array} history
 */
function updateHistoryTable(history) {
  if (typeof renderHistoryTable === 'function') {
    renderHistoryTable(history || []);
  }
}

/**
 * updateMetricsTable(metrics) — populates the ML metrics table.
 * @param {Array|null} metrics
 */
function updateMetricsTable(metrics) {
  const tbody = document.getElementById('metricsTableBody');
  if (!tbody) return;

  if (!metrics || !Array.isArray(metrics) || metrics.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-row">Model metrics not yet loaded.</td></tr>';
    return;
  }

  tbody.innerHTML = metrics.map(m => {
    const name = m.model_name || m.model || '—';
    const rmse = m.rmse != null ? Number(m.rmse).toFixed(2) : '—';
    const mae = m.mae != null ? Number(m.mae).toFixed(2) : '—';
    const r2 = m.r2 != null ? Number(m.r2).toFixed(4) : '—';
    const scoreVal = m.score != null ? m.score : (m.composite_score != null ? m.composite_score : null);
    const score = scoreVal != null ? Number(scoreVal).toFixed(3) : '—';
    const isSelected = m.selected === true || m.is_best === true;
    return `
    <tr>
      <td><strong>${escapeHtml(name.toUpperCase())}</strong></td>
      <td>${rmse}</td>
      <td>${mae}</td>
      <td>${r2}</td>
      <td>${score}</td>
      <td>${isSelected
        ? '<span class="selected-model"><i class="fa-solid fa-check-circle"></i> Selected</span>'
        : '—'}</td>
    </tr>`;
  }).join('');
}

/* ------------------------------------------------------------------ */
/*  Button handlers                                                     */
/* ------------------------------------------------------------------ */

/**
 * handleReplayPlay() — starts progressive replay streaming.
 */
async function handleReplayPlay() {
  const speed = document.getElementById('replaySpeedSelect')?.value || '1.0';
  try {
    const res = await fetch(`${API_BASE}/twin/replay/start?engine_id=${encodeURIComponent(currentEngineId)}&interval_sec=${speed}`, {
      method: 'POST',
    });
    if (res.ok) {
      const data = await res.json();
      currentTwinState = data.twin_state;
      updateAllUI(data.twin_state, data.status);
    }
  } catch (err) {
    console.error('[Digital Twin] Play error:', err);
  }
}

/**
 * handleReplayPause() — pauses progressive replay.
 */
async function handleReplayPause() {
  try {
    const res = await fetch(`${API_BASE}/twin/replay/pause?engine_id=${encodeURIComponent(currentEngineId)}`, {
      method: 'POST',
    });
    if (res.ok) {
      const data = await res.json();
      currentTwinState = data.twin_state;
      updateAllUI(data.twin_state, data.status);
    }
  } catch (err) {
    console.error('[Digital Twin] Pause error:', err);
  }
}

/**
 * handleReplayStep() — advances replay by exactly 1 cycle.
 */
async function handleReplayStep() {
  try {
    const res = await fetch(`${API_BASE}/twin/replay/step?engine_id=${encodeURIComponent(currentEngineId)}`, {
      method: 'POST',
    });
    if (res.ok) {
      const data = await res.json();
      currentTwinState = data.twin_state;
      updateAllUI(data.twin_state, data.status);
    }
  } catch (err) {
    console.error('[Digital Twin] Step error:', err);
  }
}

/**
 * handleReplayReset() — resets replay and twin state to cycle 0.
 */
async function handleReplayReset() {
  try {
    const res = await fetch(`${API_BASE}/twin/replay/reset?engine_id=${encodeURIComponent(currentEngineId)}`, {
      method: 'POST',
    });
    if (res.ok) {
      const data = await res.json();
      currentTwinState = data.twin_state;
      updateAllUI(data.twin_state, data.status);
    }
  } catch (err) {
    console.error('[Digital Twin] Reset error:', err);
  }
}

/**
 * handleReplaySpeed() — updates playback speed.
 */
async function handleReplaySpeed() {
  const speed = document.getElementById('replaySpeedSelect')?.value || '1.0';
  try {
    await fetch(`${API_BASE}/twin/replay/speed?engine_id=${encodeURIComponent(currentEngineId)}&interval_sec=${speed}`, {
      method: 'POST',
    });
  } catch (err) {
    console.error('[Digital Twin] Speed error:', err);
  }
}

/**
 * handleAnalyze() — calls the AI agent analysis endpoint.
 */
async function handleAnalyze() {
  const btn = document.getElementById('btnAnalyze');
  const thinking = document.getElementById('agentThinking');

  console.log('[Digital Twin] Analyze Engine clicked');
  console.log('[Digital Twin] Engine:', currentEngineId);
  console.log('[Digital Twin] API:', API_BASE);

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';
  }

  if (thinking) {
    thinking.style.display = 'inline-flex';
  }

  try {
    // ---------------------------------------------------------
    // 1. Get current Digital Twin state
    // ---------------------------------------------------------
    console.log('[Digital Twin] Getting twin state...');

    const stateResponse = await fetch(
      `${API_BASE}/twin/state?engine_id=${encodeURIComponent(currentEngineId)}`
    );

    if (!stateResponse.ok) {
      throw new Error(
        `Twin API failed: ${stateResponse.status} ${stateResponse.statusText}`
      );
    }

    const twinState = await stateResponse.json();

    console.log('[Digital Twin] Twin state:', twinState);

    // ---------------------------------------------------------
    // 2. Run AI Agent analysis
    // ---------------------------------------------------------
    console.log('[Digital Twin] Running AI agent analysis...');

    const analysisResponse = await fetch(
      `${API_BASE}/agent/analyze?engine_id=${encodeURIComponent(currentEngineId)}`
    );

    if (!analysisResponse.ok) {
      const errorText = await analysisResponse.text();

      throw new Error(
        `Agent API failed: ${analysisResponse.status} ${errorText}`
      );
    }

    const analysis = await analysisResponse.json();

    console.log('[Digital Twin] Analysis:', analysis);

    // Save globally so Approve button can use it
    currentAnalysis = analysis;

    // ---------------------------------------------------------
    // 3. Update RUL / health / risk section
    // ---------------------------------------------------------
    updateAllUI(twinState);

    // ---------------------------------------------------------
    // 4. Update AI Agent section
    // ---------------------------------------------------------
    updateAgentPanel(analysis);

    // ---------------------------------------------------------
    // 5. Populate What-if Simulation
    // ---------------------------------------------------------

    if (typeof populateInterventionSelect === 'function') {
      try {
        const interventionsResponse = await fetch(
          `${API_BASE}/simulation/interventions?engine_id=${encodeURIComponent(currentEngineId)}`
        );

        if (!interventionsResponse.ok) {
          throw new Error(
            `Interventions API failed: ${interventionsResponse.status}`
          );
        }

        const interventionData = await interventionsResponse.json();

        populateInterventionSelect(
          interventionData.interventions || []
        );

        console.log(
          '[Digital Twin] Loaded interventions:',
          interventionData.interventions
        );
      } catch (error) {
        console.error('[Digital Twin] Failed to load interventions:', error);
        populateInterventionSelect(analysis.interventions || []);
      }
    }

    // ---------------------------------------------------------
    // 6. Show success message
    // ---------------------------------------------------------
    showToast('AI analysis complete.', 'success');

    console.log('[Digital Twin] Analysis completed successfully.');

  } catch (err) {

    console.error(
      '[Digital Twin] Analyze Engine failed:',
      err
    );

    showToast(
      `Analysis failed: ${err.message}`,
      'error'
    );

  } finally {

    if (btn) {
      btn.disabled = false;
      btn.innerHTML =
        '<i class="fa-solid fa-robot"></i> Analyze Engine';
    }

    if (thinking) {
      thinking.style.display = 'none';
    }
  }
}
/**
 * handleApprove() — approves the current AI recommendation.
 */
async function handleApprove() {
  if (!currentAnalysis) {
    showToast('Run analysis before approving.', 'warning');
    return;
  }

  const dropdown = document.getElementById('whatIfSelect');

  const interventionId =
    (dropdown && dropdown.value)
    || (typeof selectedInterventionId !== 'undefined' && selectedInterventionId)
    || currentAnalysis.recommended_intervention_id
    || null;

  console.log('[APPROVE DEBUG] Dropdown value:', dropdown?.value);
  console.log('[APPROVE DEBUG] Sending intervention ID:', interventionId);

  const notes = (document.getElementById('engineerNotes') || {}).value || '';
  const btn = document.getElementById('btnApprove');

  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Approving…'; }

  try {
    const response = await fetch(`${API_BASE}/agent/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine_id: currentEngineId,
        intervention_id: interventionId,
        engineer_notes: notes,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const result = await response.json();
    showToast('Intervention approved. The recommended action has been logged.', 'success');
    updateApprovalStatusBadge('approved');
    fetchEngineState();
  } catch (err) {
    console.error('[Digital Twin] handleApprove error:', err);
    showToast(`Approval failed: ${err.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Approve'; }
  }
}

/**
 * handleReject() — rejects the current AI recommendation.
 */
async function handleReject() {
  if (!currentAnalysis) {
    showToast('Run analysis before rejecting.', 'warning');
    return;
  }

  const interventionId =
    (typeof selectedInterventionId !== 'undefined' && selectedInterventionId) ||
    currentAnalysis.recommended_intervention_id ||
    null;

  if (!interventionId) {
    showToast('No intervention selected. Run analysis again.', 'warning');
    return;
  }

  const notes =
    (document.getElementById('engineerNotes') || {}).value || '';

  const btn = document.getElementById('btnReject');

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Rejecting…';
  }

  try {
    const response = await fetch(`${API_BASE}/agent/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engine_id: currentEngineId,
        intervention_id: interventionId,
        engineer_notes: notes
      })
    });

    const result = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(
        result.detail || `HTTP ${response.status}`
      );
    }

    showToast(
      result.message || 'Recommendation rejected. No action will be taken.',
      'info'
    );

    updateApprovalStatusBadge('rejected');
    await fetchEngineState();

  } catch (err) {
    console.error('[Digital Twin] handleReject error:', err);
    showToast(`Rejection failed: ${err.message}`, 'error');

  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML =
        '<i class="fa-solid fa-circle-xmark"></i> Reject';
    }
  }
}
/* ------------------------------------------------------------------ */
/*  Approval helpers                                                    */
/* ------------------------------------------------------------------ */

/**
 * updateApprovalPanel(analysis, twinState) — delegates to agent.js.
 * @param {Object|null} analysis
 * @param {Object|null} twinState
 */
function updateApprovalPanel(analysis, twinState) {
  if (typeof renderApprovalPanel === 'function') {
    renderApprovalPanel(analysis, twinState);
  }
}

/**
 * updateApprovalStatusBadge(status) — shows the approval/rejection badge in the header.
 * @param {'approved'|'rejected'} status
 */
function updateApprovalStatusBadge(status) {
  const badge = document.getElementById('approvalStatusBadge');
  if (!badge) return;
  if (status === 'approved') {
    badge.textContent = '✓ Approved';
    badge.className = 'approval-status-badge approved';
  } else if (status === 'rejected') {
    badge.textContent = '✗ Rejected';
    badge.className = 'approval-status-badge rejected';
  } else {
    badge.textContent = '';
    badge.className = 'approval-status-badge';
  }
}

/* ------------------------------------------------------------------ */
/*  Connection & timestamp                                              */
/* ------------------------------------------------------------------ */

/**
 * updateConnectionStatus(isConnected) — updates the header status dot.
 * @param {boolean} isConnected
 */
function updateConnectionStatus(isConnected) {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');

  if (dot) {
    dot.className = `status-dot ${isConnected ? 'connected' : 'disconnected'}`;
  }
  if (label) {
    label.textContent = isConnected ? 'Live' : 'Disconnected';
  }
}

/**
 * updateLastUpdated() — writes the current time into the header.
 */
function updateLastUpdated() {
  const el = document.getElementById('lastUpdated');
  if (el) el.textContent = formatTimestamp(new Date().toISOString());
}

/* ------------------------------------------------------------------ */
/*  Formatters                                                          */
/* ------------------------------------------------------------------ */

/**
 * formatHealthState(state) — returns {text, className} for health state.
 * @param {string} state
 * @returns {{text: string, className: string}}
 */
function formatHealthState(state) {
  const map = {
    healthy: { text: 'Healthy', className: 'healthy' },
    warning: { text: 'Warning', className: 'warning' },
    critical: { text: 'Critical', className: 'critical' },
    failure: { text: 'Failure', className: 'failure' },
  };
  const key = (state || '').toLowerCase();
  return map[key] || { text: state || 'Unknown', className: 'unknown' };
}

/**
 * formatRisk(risk) — returns {text, className, subtext} for risk level.
 * @param {string} risk
 * @returns {{text: string, className: string, subtext: string}}
 */
function formatRisk(risk) {
  const map = {
    low: { text: 'Low', className: 'risk-low', subtext: 'No immediate concern' },
    medium: { text: 'Medium', className: 'risk-medium', subtext: 'Monitor closely' },
    high: { text: 'High', className: 'risk-high', subtext: 'Maintenance recommended' },
    critical: { text: 'Critical', className: 'risk-critical', subtext: 'Immediate intervention needed' },
  };
  const key = (risk || '').toLowerCase();
  return map[key] || { text: risk || 'Unknown', className: '', subtext: '' };
}

/**
 * formatTimestamp(isoString) — converts ISO string to a readable local time.
 * @param {string} isoString
 * @returns {string}
 */
function formatTimestamp(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return isoString;
  }
}

/* ------------------------------------------------------------------ */
/*  Colour helpers                                                      */
/* ------------------------------------------------------------------ */

/**
 * getRULColor(rul) — returns a CSS colour based on remaining useful life.
 * @param {number|null} rul
 * @returns {string}
 */
function getRULColor(rul) {
  if (rul == null) return '';
  if (rul > 75) return 'var(--green)';
  if (rul > 40) return 'var(--yellow)';
  if (rul > 15) return 'var(--orange)';
  return 'var(--red)';
}

/**
 * getHealthScoreColor(score) — CSS colour for the health score value.
 * @param {number} score
 * @returns {string}
 */
function getHealthScoreColor(score) {
  if (score > 75) return 'var(--green)';
  if (score > 40) return 'var(--yellow)';
  if (score > 15) return 'var(--orange)';
  return 'var(--red)';
}

/**
 * getHealthScoreGradient(score) — gradient stops for the health bar.
 * @param {number} score
 * @returns {string}
 */
function getHealthScoreGradient(score) {
  if (score > 75) return 'var(--blue), var(--green)';
  if (score > 40) return 'var(--blue), var(--yellow)';
  if (score > 15) return 'var(--yellow), var(--orange)';
  return 'var(--orange), var(--red)';
}

/* ------------------------------------------------------------------ */
/*  Toast                                                               */
/* ------------------------------------------------------------------ */

/**
 * showToast(message, type) — creates a floating notification.
 * @param {string} message
 * @param {'success'|'error'|'info'|'warning'} type
 */
function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const iconMap = {
    success: 'fa-circle-check',
    error: 'fa-circle-xmark',
    info: 'fa-circle-info',
    warning: 'fa-triangle-exclamation',
  };

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="fa-solid ${iconMap[type] || 'fa-circle-info'}"></i>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  // Auto-dismiss after 4 s
  setTimeout(() => {
    toast.style.animation = 'toast-out 0.3s ease forwards';
    setTimeout(() => toast.remove(), 310);
  }, 4000);
}

/* ------------------------------------------------------------------ */
/*  Utility                                                             */
/* ------------------------------------------------------------------ */

/**
 * resetAgentPanel() — clears the agent panel when the engine changes.
 */
function resetAgentPanel() {
  const ids = ['agentProblem', 'agentExplanation'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<p class="placeholder-text">Run analysis to populate this section.</p>';
  });
  const indicators = document.getElementById('degradationIndicators');
  if (indicators) indicators.innerHTML = '<li class="placeholder-text">No data — run analysis.</li>';
  const steps = document.getElementById('reasoningSteps');
  if (steps) steps.innerHTML = '<li class="placeholder-text">Reasoning steps will appear here after analysis.</li>';
}

/**
 * escapeHtml(str) — basic HTML-entity escaping for user-facing strings.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str ?? '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ------------------------------------------------------------------ */
/*  Bootstrap                                                           */
/* ------------------------------------------------------------------ */
document.addEventListener('DOMContentLoaded', init);

