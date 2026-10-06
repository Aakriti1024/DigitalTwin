/**
 * charts.js — Digital Twin Dashboard
 * Manages all Chart.js instances: RUL trend, health gauge, sensor trends.
 */

'use strict';

/* ------------------------------------------------------------------ */
/*  Module-level state                                                  */
/* ------------------------------------------------------------------ */
let rulTrendChart    = null;
let healthGaugeChart = null;
let sensorTrendChart = null;

/** Active sensors displayed on the sensor trend chart. */
let activeSensors = ['s2', 's3', 's4', 's7', 's9', 's11'];

/* ------------------------------------------------------------------ */
/*  Color palettes                                                      */
/* ------------------------------------------------------------------ */

/** Color mapping for sensor lines. */
const SENSOR_COLORS = {
  s2:  '#3b82f6',  // blue
  s3:  '#22c55e',  // green
  s4:  '#eab308',  // yellow
  s5:  '#f97316',  // orange
  s6:  '#ec4899',  // pink
  s7:  '#8b5cf6',  // purple
  s8:  '#06b6d4',  // cyan
  s9:  '#f43f5e',  // rose
  s10: '#10b981',  // emerald
  s11: '#a78bfa',  // violet
  s12: '#fb923c',  // light-orange
  s13: '#34d399',  // teal
  s14: '#60a5fa',  // light-blue
  s15: '#fbbf24',  // amber
  s16: '#c084fc',  // light-purple
  s17: '#4ade80',  // light-green
  s18: '#f87171',  // light-red
  s19: '#38bdf8',  // sky
  s20: '#fdba74',  // peach
  s21: '#a5f3fc',  // light-cyan
};

/** Sensor human-readable labels. */
const SENSOR_LABELS = {
  s2:  'S2 Fan inlet temp',
  s3:  'S3 LPC outlet temp',
  s4:  'S4 HPC outlet temp',
  s5:  'S5 LPT outlet temp',
  s6:  'S6 Total pressure ratio',
  s7:  'S7 HPC outlet pressure',
  s8:  'S8 Physical fan speed',
  s9:  'S9 Engine pressure ratio',
  s10: 'S10 Static pressure ratio',
  s11: 'S11 HPC outlet static pressure',
  s12: 'S12 Fuel flow ratio',
  s13: 'S13 Corrected fan speed',
  s14: 'S14 BPR',
  s15: 'S15 Bleed enthalpy',
  s16: 'S16 HPT coolant bleed',
  s17: 'S17 LPT coolant bleed',
  s18: 'S18 Bypass ratio',
  s19: 'S19 Burner fuel–air ratio',
  s20: 'S20 HPC efficiency ratio',
  s21: 'S21 LPT efficiency ratio',
};

/* ------------------------------------------------------------------ */
/*  Global Chart.js defaults (dark theme)                              */
/* ------------------------------------------------------------------ */

/**
 * CHART_DEFAULTS — dark theme defaults applied globally to Chart.js.
 */
const CHART_DEFAULTS = {
  color:         '#94a3b8',
  borderColor:   '#1e3050',
  backgroundColor:'#1a2236',
};

// Apply global defaults
if (typeof Chart !== 'undefined') {
  Chart.defaults.color            = CHART_DEFAULTS.color;
  Chart.defaults.borderColor      = CHART_DEFAULTS.borderColor;
  Chart.defaults.backgroundColor  = CHART_DEFAULTS.backgroundColor;
  Chart.defaults.font.family      = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size        = 12;
  Chart.defaults.animation.duration = 400;
}

/* ------------------------------------------------------------------ */
/*  Initialization                                                      */
/* ------------------------------------------------------------------ */

/**
 * initCharts() — creates all chart instances.
 * Called once from main.js init().
 */
function initCharts() {
  createRULTrendChart();
  createHealthGaugeChart();
  createSensorTrendChart();
}

/* ================================================================
   RUL TREND CHART
   ================================================================ */

/**
 * createRULTrendChart() — creates the line chart that shows the last
 * 50 cycles of predicted RUL with upper/lower uncertainty bands.
 */
function createRULTrendChart() {
  const canvas = document.getElementById('rulTrendChart');
  if (!canvas) return;
  if (rulTrendChart) { rulTrendChart.destroy(); rulTrendChart = null; }

  const ctx = canvas.getContext('2d');

  rulTrendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels:   [],
      datasets: [
        // Upper bound (hidden fill target)
        {
          label:           'Upper Bound',
          data:            [],
          borderColor:     'transparent',
          backgroundColor: 'rgba(59,130,246,0.1)',
          fill:            '+1',  // fill to next dataset
          pointRadius:     0,
          tension:         0.3,
        },
        // Main RUL line
        {
          label:           'Predicted RUL',
          data:            [],
          borderColor:     '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.08)',
          borderWidth:     2.5,
          pointRadius:     3,
          pointHoverRadius: 5,
          pointBackgroundColor: '#3b82f6',
          tension:         0.3,
          fill:            false,
        },
        // Lower bound
        {
          label:           'Lower Bound',
          data:            [],
          borderColor:     'transparent',
          backgroundColor: 'rgba(59,130,246,0.1)',
          fill:            '-1',  // fill to previous dataset
          pointRadius:     0,
          tension:         0.3,
        },
      ],
    },
    options: {
      responsive:         true,
      maintainAspectRatio: false,
      interaction: {
        mode:         'index',
        intersect:    false,
      },
      plugins: {
        legend: {
          display:  true,
          position: 'top',
          labels: {
            color:    '#94a3b8',
            boxWidth: 14,
            padding:  14,
            filter: (item) => item.text !== 'Upper Bound' && item.text !== 'Lower Bound',
          },
        },
        tooltip: {
          backgroundColor: 'rgba(17,24,39,0.95)',
          borderColor:     '#1e3050',
          borderWidth:     1,
          titleColor:      '#f1f5f9',
          bodyColor:       '#94a3b8',
          callbacks: {
            label: (ctx) => {
              if (ctx.dataset.label === 'Upper Bound' || ctx.dataset.label === 'Lower Bound') {
                return null;
              }
              return ` RUL: ${Math.round(ctx.parsed.y)} cycles`;
            },
          },
        },
        // Threshold annotations (drawn as horizontal lines via a custom plugin)
        // NOTE: chart.js-plugin-annotation is not loaded via CDN in this file;
        //       thresholds are drawn as reference lines below using a custom inline plugin.
      },
      scales: {
        x: {
          title: {
            display: true,
            text:    'Engine Cycle',
            color:   '#475569',
          },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#475569', maxTicksLimit: 10 },
        },
        y: {
          title: {
            display: true,
            text:    'Remaining Useful Life (cycles)',
            color:   '#475569',
          },
          min:   0,
          max:   130,
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#475569' },
        },
      },
    },
    // Inline plugin: draw threshold reference lines
    plugins: [{
      id: 'thresholdLines',
      afterDraw(chart) {
        const { ctx, chartArea, scales } = chart;
        if (!chartArea) return;

        const thresholds = [
          { value: 75, color: 'rgba(34,197,94,0.5)',   label: 'Healthy' },
          { value: 40, color: 'rgba(234,179,8,0.5)',   label: 'Warning' },
          { value: 15, color: 'rgba(239,68,68,0.55)',  label: 'Critical' },
        ];

        ctx.save();
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.2;

        thresholds.forEach(({ value, color, label }) => {
          const y = scales.y.getPixelForValue(value);
          if (y < chartArea.top || y > chartArea.bottom) return;

          ctx.strokeStyle = color;
          ctx.beginPath();
          ctx.moveTo(chartArea.left, y);
          ctx.lineTo(chartArea.right, y);
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.font = '10px Inter, sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(label, chartArea.left + 4, y - 4);
        });

        ctx.restore();
      },
    }],
  });
}

/**
 * updateRULTrendChart(historyData) — pushes new data into the RUL trend chart.
 * @param {Array<{cycle: number, rul: number, rul_lower?: number, rul_upper?: number}>} historyData
 */
function updateRULTrendChart(historyData) {
  if (!rulTrendChart || !Array.isArray(historyData)) return;

  const labels  = historyData.map(d => d.cycle);
  const rul     = historyData.map(d => d.rul     != null ? Math.round(d.rul)       : null);
  const upper   = historyData.map(d => d.rul_upper != null ? Math.round(d.rul_upper) : null);
  const lower   = historyData.map(d => d.rul_lower != null ? Math.round(d.rul_lower) : null);

  // If bounds missing, remove uncertainty bands
  const hasBands = upper.some(v => v != null) && lower.some(v => v != null);

  rulTrendChart.data.labels = labels;
  rulTrendChart.data.datasets[0].data = hasBands ? upper : [];
  rulTrendChart.data.datasets[1].data = rul;
  rulTrendChart.data.datasets[2].data = hasBands ? lower : [];

  // Dynamically colour the RUL line based on the latest value
  const latestRUL = rul[rul.length - 1];
  if (latestRUL != null) {
    const lineColor = latestRUL > 75 ? '#3b82f6'
                    : latestRUL > 40 ? '#eab308'
                    : latestRUL > 15 ? '#f97316'
                    :                  '#ef4444';
    rulTrendChart.data.datasets[1].borderColor          = lineColor;
    rulTrendChart.data.datasets[1].pointBackgroundColor = lineColor;
  }

  rulTrendChart.update();
}

/* ================================================================
   HEALTH GAUGE CHART
   ================================================================ */

/**
 * createHealthGaugeChart() — creates the doughnut health gauge.
 */
function createHealthGaugeChart() {
  const canvas = document.getElementById('healthGaugeChart');
  if (!canvas) return;
  if (healthGaugeChart) { healthGaugeChart.destroy(); healthGaugeChart = null; }

  const ctx = canvas.getContext('2d');

  healthGaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data:            [0, 100],
        backgroundColor: ['#22c55e', 'rgba(255,255,255,0.04)'],
        borderWidth:     0,
        borderRadius:    6,
        hoverOffset:     4,
      }],
    },
    options: {
      responsive:         true,
      maintainAspectRatio: false,
      cutout:             '76%',
      plugins: {
        legend:  { display: false },
        tooltip: { enabled: false },
      },
      animation: {
        animateRotate: true,
        duration:      600,
      },
    },
  });
}

/**
 * updateHealthGaugeChart(healthScore, healthState) — updates the gauge.
 * @param {number} healthScore  0-100
 * @param {string} healthState  'healthy'|'warning'|'critical'|'failure'
 */
function updateHealthGaugeChart(healthScore, healthState) {
  if (!healthGaugeChart) return;

  const score = Math.min(100, Math.max(0, healthScore || 0));

  // Choose colour by score
  let color;
  if (score > 75)      color = '#22c55e';  // green
  else if (score > 40) color = '#eab308';  // yellow
  else if (score > 15) color = '#f97316';  // orange
  else                 color = '#ef4444';  // red

  healthGaugeChart.data.datasets[0].data            = [score, 100 - score];
  healthGaugeChart.data.datasets[0].backgroundColor = [color, 'rgba(255,255,255,0.04)'];
  healthGaugeChart.update();

  // Update the center text
  const centerVal = document.getElementById('gaugeCenterValue');
  if (centerVal) {
    centerVal.textContent = Math.round(score);
    centerVal.style.color = color;
  }
}

/* ================================================================
   SENSOR TREND CHART
   ================================================================ */

/**
 * createSensorTrendChart() — creates the multi-line sensor history chart.
 */
function createSensorTrendChart() {
  const canvas = document.getElementById('sensorTrendChart');
  if (!canvas) return;
  if (sensorTrendChart) { sensorTrendChart.destroy(); sensorTrendChart = null; }

  const ctx = canvas.getContext('2d');

  sensorTrendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels:   [],
      datasets: activeSensors.map(sensorId => buildSensorDataset(sensorId, [])),
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction: {
        mode:      'index',
        intersect: false,
      },
      plugins: {
        legend: {
          display:  true,
          position: 'top',
          labels: {
            color:    '#94a3b8',
            boxWidth: 12,
            padding:  12,
          },
        },
        tooltip: {
          backgroundColor: 'rgba(17,24,39,0.95)',
          borderColor:     '#1e3050',
          borderWidth:     1,
          titleColor:      '#f1f5f9',
          bodyColor:       '#94a3b8',
          callbacks: {
            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}`,
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text:    'Engine Cycle',
            color:   '#475569',
          },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#475569', maxTicksLimit: 10 },
        },
        y: {
          title: {
            display: true,
            text:    'Normalized Value (0–1)',
            color:   '#475569',
          },
          min:   0,
          max:   1,
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#475569' },
        },
      },
    },
  });
}

/**
 * buildSensorDataset(sensorId, data) — constructs a Chart.js dataset config.
 * @param {string} sensorId  e.g. 's2'
 * @param {number[]} data    Normalized values.
 * @returns {Object}
 */
function buildSensorDataset(sensorId, data) {
  const color = SENSOR_COLORS[sensorId] || '#94a3b8';
  return {
    label:            SENSOR_LABELS[sensorId] || sensorId.toUpperCase(),
    data,
    borderColor:      color,
    backgroundColor:  hexToRgba(color, 0.08),
    borderWidth:      1.8,
    pointRadius:      1.5,
    pointHoverRadius: 4,
    tension:          0.35,
    fill:             false,
    sensorId,  // custom property for toggle lookup
  };
}

/**
 * updateSensorTrendChart(historyData, sensors) — refreshes sensor data.
 * @param {Array<{cycle: number, sensors: Object}>} historyData
 * @param {string[]} [sensors]  Optional override of displayed sensors.
 */
function updateSensorTrendChart(historyData, sensors) {
  if (!sensorTrendChart || !Array.isArray(historyData)) return;

  const displaySensors = sensors || activeSensors;

  // Build min/max lookup for normalization (per sensor over the visible window)
  const minMax = {};
  displaySensors.forEach(sid => {
    const values = historyData
      .map(d => d.sensors && d.sensors[sid] != null ? d.sensors[sid] : null)
      .filter(v => v != null);
    minMax[sid] = {
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 1,
    };
  });

  const normalize = (value, sid) => {
    const { min, max } = minMax[sid];
    if (max === min) return 0.5;
    return (value - min) / (max - min);
  };

  const labels = historyData.map(d => d.cycle);

  sensorTrendChart.data.labels   = labels;
  sensorTrendChart.data.datasets = displaySensors.map(sid => {
    const values = historyData.map(d =>
      d.sensors && d.sensors[sid] != null ? normalize(d.sensors[sid], sid) : null
    );
    return buildSensorDataset(sid, values);
  });

  sensorTrendChart.update();
}

/**
 * toggleSensor(sensorId) — adds or removes a sensor from the active list
 * and immediately refreshes the chart with the last known history (no new fetch).
 * @param {string} sensorId
 */
function toggleSensor(sensorId) {
  const idx = activeSensors.indexOf(sensorId);
  if (idx === -1) {
    activeSensors.push(sensorId);
  } else {
    activeSensors.splice(idx, 1);
  }

  // Re-render with existing chart data if available
  if (!sensorTrendChart) return;
  const existing = sensorTrendChart.data.labels;

  if (existing && existing.length > 0) {
    // Rebuild datasets from current chart data
    sensorTrendChart.data.datasets = activeSensors.map(sid => {
      const existing_ds = sensorTrendChart.data.datasets.find(ds => ds.sensorId === sid);
      return buildSensorDataset(sid, existing_ds ? existing_ds.data : []);
    });
    sensorTrendChart.update();
  }
}

/* ------------------------------------------------------------------ */
/*  Utility                                                             */
/* ------------------------------------------------------------------ */

/**
 * hexToRgba(hex, alpha) — converts a hex colour string to rgba().
 * @param {string} hex
 * @param {number} alpha  0–1
 * @returns {string}
 */
function hexToRgba(hex, alpha = 1) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return `rgba(148,163,184,${alpha})`;
  const r = parseInt(result[1], 16);
  const g = parseInt(result[2], 16);
  const b = parseInt(result[3], 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
