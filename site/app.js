/* Static dashboard: reads precomputed JSON from ./data and renders with Plotly. */
"use strict";

const SEV = ["low", "medium", "high", "critical"];
const MODEL_LABEL = {
  seasonal_naive: "Seasonal naive",
  lightgbm_global: "LightGBM (global)",
  sarima: "SARIMA",
  prophet: "Prophet",
};
const state = { data: {}, cellIndex: new Map(), seriesCache: new Map(), selectedCell: null, selectedMetric: "internet" };

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const fmtInt = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString());
const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(d)}%`);
const shortTs = (iso) => (iso ? new Date(iso).toISOString().slice(5, 16).replace("T", " ") : "–");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

function layout(extra = {}) {
  const ink = css("--ink"), muted = css("--muted"), line = css("--line");
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: ink, size: 12, family: getComputedStyle(document.body).fontFamily },
    margin: { l: 48, r: 12, t: 8, b: 40 },
    xaxis: { gridcolor: line, zerolinecolor: line, color: muted },
    yaxis: { gridcolor: line, zerolinecolor: line, color: muted },
    legend: { orientation: "h", y: -0.18 },
    hoverlabel: { bgcolor: css("--card"), bordercolor: line, font: { color: ink } },
    ...extra,
  };
}
const CONFIG = { displayModeBar: false, responsive: true };
const palette = () => [css("--accent"), "#f59e0b", "#10b981", "#a855f7", "#ef4444"];

function table(el, cols, rows, onClick) {
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.label)}</th>`).join("");
  const body = rows
    .map((r, i) => `<tr class="${onClick ? "click" : ""}" data-i="${i}">${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.render ? c.render(r) : esc(r[c.key] ?? "–")}</td>`).join("")}</tr>`)
    .join("");
  el.innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body || `<tr><td colspan="${cols.length}" class="hint">nothing to show</td></tr>`}</tbody></table></div>`;
  if (onClick) el.querySelectorAll("tr.click").forEach((tr) => tr.addEventListener("click", () => onClick(rows[+tr.dataset.i])));
}

const sevBadge = (s) => `<span class="sev ${s}">${s}</span>`;
const ALERT_COLS = [
  { label: "#", key: "event_id", num: true },
  { label: "Severity", render: (r) => sevBadge(r.severity) },
  { label: "Cell", key: "cell_id", num: true },
  { label: "Metric", key: "metric" },
  { label: "Start (UTC)", render: (r) => shortTs(r.start) },
  { label: "Hours", key: "n_hours", num: true },
  { label: "Observed", render: (r) => fmtInt(r.observed), num: true },
  { label: "Expected", render: (r) => fmtInt(r.expected), num: true },
  { label: "Δ", render: (r) => `${r.deviation_pct > 0 ? "+" : ""}${fmt(r.deviation_pct, 0)}%`, num: true },
  { label: "Score", render: (r) => fmt(r.score, 2), num: true },
];

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    const names = ["meta", "cells", "eda", "alerts", "models", "capacity", "monitoring"];
    const loaded = await Promise.all(names.map((n) => getJSON(`data/${n}.json`)));
    names.forEach((n, i) => (state.data[n] = loaded[i]));
  } catch (e) {
    $("#subtitle").textContent = `No data yet - run the pipeline (${e.message})`;
    return;
  }
  const { meta, cells } = state.data;
  cells.cells.forEach((c, i) => state.cellIndex.set(c.cell_id, { ...c, i }));
  $("#subtitle").textContent = `${meta.source} data · ${meta.n_cells} cells · ${meta.data_start.slice(0, 10)} → ${meta.data_end.slice(0, 10)} · ${meta.timezone}`;
  $("#generated").textContent = `generated ${meta.generated_at.slice(0, 16).replace("T", " ")} UTC`;

  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1) || "overview", false));

  renderOverview();
  setupCellExplorer();
  renderAlerts();
  showTab(location.hash.slice(1) || "overview", false);
}

const rendered = new Set();
function showTab(name, push = true) {
  if (!document.getElementById(name)) name = "overview";
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (push) history.replaceState(null, "", `#${name}`);
  // plotly needs visible containers: render lazily on first show
  if (!rendered.has(name)) {
    rendered.add(name);
    ({ map: renderMap, models: renderModels, capacity: renderCapacity, monitoring: renderMonitoring, cell: () => loadCell() }[name] || (() => {}))();
  }
  document.querySelectorAll(`#${name} .js-plotly-plot`).forEach((el) => Plotly.Plots.resize(el));
}

/* ------------------------------------------------------------------ overview */
function renderOverview() {
  const { meta, alerts, models, capacity, eda } = state.data;
  const ov = Object.fromEntries((models.overall || []).map((r) => [r.model, r]));
  const lgb = ov.lightgbm_global || {}, sn = ov.seasonal_naive || {};
  const ev = alerts.evaluation?.event_level;
  const capOver = capacity.cells.filter((c) => c.status === "over_capacity" || c.status === "at_risk").length;
  const kpis = [
    ["Cells monitored", fmtInt(meta.n_cells)],
    ["Forecast MASE (LightGBM)", fmt(lgb.mase, 3)],
    ["vs. seasonal naive", sn.mase ? `${fmt(100 * (1 - lgb.mase / sn.mase), 1)}% better` : "–"],
    [`${Math.round(meta.interval_level * 100)}% interval coverage`, pct(lgb.coverage)],
    ["Active alerts", fmtInt(alerts.alerts.length)],
    ["Critical", fmtInt(alerts.alerts.filter((a) => a.severity === "critical").length)],
    ["Capacity at risk / over", fmtInt(capOver)],
    ...(ev ? [["Alert precision / recall", `${pct(ev.precision, 0)} / ${pct(ev.recall, 0)}`]] : []),
  ];
  $("#kpis").innerHTML = kpis.map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${esc(l)}</div></div>`).join("");

  const dt = eda.daily_totals;
  const traces = meta.metrics.map((m, i) => ({
    x: dt.dates, y: dt[m], name: m, type: "scatter", mode: "lines",
    yaxis: m === "internet" ? "y2" : "y", line: { width: 2, color: palette()[i] },
  }));
  Plotly.newPlot("daily", traces, layout({
    yaxis2: { overlaying: "y", side: "right", showgrid: false, color: css("--muted"), title: { text: "internet" } },
    yaxis: { ...layout().yaxis, title: { text: "sms / calls" } },
    margin: { l: 56, r: 56, t: 8, b: 40 },
  }), CONFIG);

  const sel = $("#heat-metric");
  sel.innerHTML = meta.metrics.map((m) => `<option ${m === "internet" ? "selected" : ""}>${m}</option>`).join("");
  sel.addEventListener("change", drawHeat);
  drawHeat();

  table($("#top-alerts"), ALERT_COLS, alerts.alerts.slice(0, 8), openAlert);
}

function drawHeat() {
  const m = $("#heat-metric").value;
  const rows = state.data.eda.area_how_heatmap[m];
  const areas = Object.keys(rows);
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const x = Array.from({ length: 168 }, (_, k) => `${days[Math.floor(k / 24)]} ${String(k % 24).padStart(2, "0")}h`);
  Plotly.newPlot("heat", [{
    z: areas.map((a) => rows[a]), x, y: areas, type: "heatmap", colorscale: "Viridis",
    hovertemplate: "%{y} · %{x}<br>%{z:.2f}× cell mean<extra></extra>", colorbar: { thickness: 10 },
  }], layout({ margin: { l: 90, r: 10, t: 8, b: 50 }, xaxis: { ...layout().xaxis, nticks: 14 } }), CONFIG);
}

/* ------------------------------------------------------------------ map */
function renderMap() {
  const { cells } = state.data;
  const list = cells.cells;
  const rc = list.map((c) => ({ row: c.row ?? Math.floor((c.cell_id - 1) / 100), col: c.col ?? (c.cell_id - 1) % 100 }));
  const draw = () => {
    const h = +$("#hour").value;
    $("#hour-label").textContent = `${String(h).padStart(2, "0")}:00`;
    const z = list.map((_, i) => cells.hourly_mean_internet[i][h]);
    const alerted = list.map((c, i) => [c, i]).filter(([c]) => c.alerts > 0);
    Plotly.react("gridmap", [
      {
        x: rc.map((p) => p.col), y: rc.map((p) => p.row), mode: "markers", type: "scattergl",
        marker: { symbol: "square", size: 11, color: z.map((v) => Math.log10(1 + (v || 0))), colorscale: "Viridis",
          colorbar: { title: { text: "log10 MB" }, thickness: 10 } },
        text: list.map((c, i) => `cell ${c.cell_id} · ${c.area_type ?? ""}<br>${fmtInt(z[i])} at ${h}:00 · ${c.alerts} alerts`),
        hoverinfo: "text", customdata: list.map((c) => c.cell_id), name: "cells",
      },
      {
        x: alerted.map(([, i]) => rc[i].col), y: alerted.map(([, i]) => rc[i].row), mode: "markers", type: "scatter",
        marker: { symbol: "circle-open", size: 15, line: { width: 2 }, color: css("--bad") },
        hoverinfo: "skip", name: "alerts", customdata: alerted.map(([c]) => c.cell_id),
      },
    ], layout({
      xaxis: { visible: false, scaleanchor: "y" }, yaxis: { visible: false, autorange: "reversed" },
      margin: { l: 8, r: 8, t: 8, b: 8 }, showlegend: false,
    }), CONFIG);
  };
  $("#hour").addEventListener("input", draw);
  draw();
  $("#gridmap").on("plotly_click", (ev) => {
    const id = ev.points?.[0]?.customdata;
    if (id) openCell(id);
  });
}

/* ------------------------------------------------------------------ cell explorer */
function setupCellExplorer() {
  const { cells, meta } = state.data;
  const withSeries = cells.cells.filter((c) => c.has_series).sort((a, b) => b.alerts - a.alerts || (b.mean_internet ?? 0) - (a.mean_internet ?? 0));
  $("#cell-select").innerHTML = withSeries
    .map((c) => `<option value="${c.cell_id}">${c.cell_id} · ${c.area_type ?? "cell"}${c.alerts ? ` · ${c.alerts}⚠` : ""}</option>`)
    .join("");
  $("#metric-select").innerHTML = meta.metrics.map((m) => `<option ${m === "internet" ? "selected" : ""}>${m}</option>`).join("");
  state.selectedCell = withSeries[0]?.cell_id ?? null;
  $("#cell-select").addEventListener("change", (e) => { state.selectedCell = +e.target.value; loadCell(); });
  $("#metric-select").addEventListener("change", (e) => { state.selectedMetric = e.target.value; loadCell(); });
}

function openCell(cellId, metric) {
  const c = state.cellIndex.get(cellId);
  if (!c?.has_series) {
    alert(`Cell ${cellId} has no exported series (only top / alerted / capacity-flagged cells are exported).`);
    return;
  }
  state.selectedCell = cellId;
  if (metric) state.selectedMetric = metric;
  $("#cell-select").value = String(cellId);
  $("#metric-select").value = state.selectedMetric;
  showTab("cell");
  loadCell();
}

function openAlert(a) { openCell(a.cell_id, a.metric); }

async function loadCell() {
  const id = state.selectedCell, m = state.selectedMetric;
  if (!id) return;
  let doc = state.seriesCache.get(id);
  if (!doc) {
    doc = await getJSON(`data/series/${id}.json`);
    state.seriesCache.set(id, doc);
  }
  const c = state.cellIndex.get(id) || {};
  $("#cell-meta").textContent = [c.area_type, c.capacity_internet ? `capacity ${fmtInt(c.capacity_internet)}` : null, c.capacity_status]
    .filter(Boolean).join(" · ");
  const s = doc.metrics[m];
  const x = doc.history_ts;
  const accent = css("--accent"), muted = css("--muted"), bad = css("--bad"), fc = "#f59e0b";
  const band = (xs, lo, hi, color, name) => [
    { x: xs, y: hi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, type: "scatter" },
    { x: xs, y: lo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: color, name, type: "scatter", hoverinfo: "skip" },
  ];
  const traces = [];
  if (s.expected) traces.push(...band(x, s.lo, s.hi, "rgba(100,116,139,0.18)", "expected band"));
  traces.push({ x, y: s.actual, name: "actual", mode: "lines", line: { color: accent, width: 1.8 }, type: "scatter" });
  if (s.expected) traces.push({ x, y: s.expected, name: "expected", mode: "lines", line: { color: muted, width: 1.2, dash: "dot" }, type: "scatter" });
  traces.push(...band(s.forecast_ts, s.forecast_lo, s.forecast_hi, "rgba(245,158,11,0.18)", "forecast interval"));
  traces.push({ x: s.forecast_ts, y: s.forecast, name: "forecast", mode: "lines", line: { color: fc, width: 2 }, type: "scatter" });

  const cellAlerts = state.data.alerts.alerts.filter((a) => a.cell_id === id);
  const shapes = cellAlerts.filter((a) => a.metric === m).map((a) => ({
    type: "rect", xref: "x", yref: "paper", x0: a.start, x1: a.end, y0: 0, y1: 1,
    fillcolor: bad, opacity: 0.12, line: { width: 0 },
  }));
  if (m === "internet" && c.capacity_internet) {
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: c.capacity_internet, y1: c.capacity_internet, line: { color: bad, dash: "dash", width: 1 } });
  }
  Plotly.react("cell-chart", traces, layout({ shapes, hovermode: "x unified" }), CONFIG);
  table($("#cell-alerts"), ALERT_COLS, cellAlerts, (a) => { state.selectedMetric = a.metric; $("#metric-select").value = a.metric; loadCell(); });
}

/* ------------------------------------------------------------------ alerts */
function renderAlerts() {
  const { alerts, meta } = state.data;
  $("#alert-metric").innerHTML += meta.metrics.map((m) => `<option>${m}</option>`).join("");
  const draw = () => {
    const min = +$("#sev-filter").value, met = $("#alert-metric").value;
    const rows = alerts.alerts.filter((a) => SEV.indexOf(a.severity) >= min && (!met || a.metric === met));
    $("#alert-count").textContent = `${rows.length} alerts · ${new Set(rows.map((r) => r.incident_id)).size} incidents`;
    table($("#alerts-table"), [...ALERT_COLS, { label: "Incident", key: "incident_id", num: true }], rows.slice(0, 500), openAlert);
  };
  $("#sev-filter").addEventListener("change", draw);
  $("#alert-metric").addEventListener("change", draw);
  draw();

  const ev = alerts.evaluation;
  if (!ev) { $("#eval-card").style.display = "none"; return; }
  const e = ev.event_level;
  const point = Object.entries(ev.point_level).map(([k, v]) => ({ detector: k, ...v }));
  $("#eval").innerHTML = `
    <div class="kpis">
      ${[["Event precision", pct(e.precision, 0)], ["Event recall", pct(e.recall, 0)], ["Event F1", fmt(e.f1, 2)],
        ["Mean time to detect", `${fmt(e.mean_time_to_detect_h, 1)} h`], ["Alerts / day", fmt(e.alerts_per_day, 1)],
        ["Alerts on scheduled events", fmtInt(e.alerts_during_scheduled_events ?? 0)]]
        .map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("")}
    </div><div id="eval-points"></div><p class="hint">Recall by type: ${Object.entries(ev.by_type).map(([k, v]) => `${k} ${pct(v, 0)}`).join(" · ")}</p>`;
  table($("#eval-points"), [
    { label: "Detector (point level)", key: "detector" },
    { label: "PR-AUC", render: (r) => fmt(r.pr_auc, 3), num: true },
    { label: "Precision", render: (r) => pct(r.precision), num: true },
    { label: "Recall", render: (r) => pct(r.recall), num: true },
    { label: "F1", render: (r) => fmt(r.f1, 3), num: true },
    { label: "Flagged points", render: (r) => fmtInt(r.flagged), num: true },
  ], point);
}

/* ------------------------------------------------------------------ models */
function renderModels() {
  const { models } = state.data;
  const byH = models.by_horizon || [];
  const modelsList = [...new Set(byH.map((r) => r.model))];
  const metrics = [...new Set(byH.map((r) => r.metric))];
  const horizons = ["1h", "24h", "168h"];
  const traces = modelsList.flatMap((mod, i) => horizons.map((h, j) => ({
    type: "bar", name: `${MODEL_LABEL[mod] || mod}`, legendgroup: mod, showlegend: j === 0,
    x: metrics.map((m) => `${m}<br>${h}`), y: metrics.map((m) => byH.find((r) => r.model === mod && r.metric === m && r.horizon === h)?.mase),
    marker: { color: palette()[i] }, offsetgroup: mod, hovertemplate: "%{x}: MASE %{y:.3f}<extra>" + mod + "</extra>",
  })));
  Plotly.newPlot("mase-h", traces, layout({
    barmode: "group",
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { dash: "dot", color: css("--muted") } }],
    yaxis: { ...layout().yaxis, title: { text: "MASE (lower is better)" } },
  }), CONFIG);

  const metricCols = [
    { label: "Model", render: (r) => MODEL_LABEL[r.model] || r.model },
    { label: "MAE", render: (r) => fmt(r.mae, 2), num: true },
    { label: "RMSE", render: (r) => fmt(r.rmse, 2), num: true },
    { label: "sMAPE", render: (r) => `${fmt(r.smape, 1)}%`, num: true },
    { label: "MASE", render: (r) => fmt(r.mase, 3), num: true },
    { label: "Coverage", render: (r) => pct(r.coverage), num: true },
  ];
  table($("#overall"), metricCols, models.overall);
  const sample = models.sample_comparison || [];
  if (sample.length) {
    table($("#sample"), [{ label: "Horizon bucket", key: "bucket" }, ...metricCols], sample);
  } else {
    $("#sample").innerHTML = `<p class="hint">Statistical models were skipped in this run.</p>`;
  }
  const t = models.timings || {};
  $("#sample").insertAdjacentHTML("beforeend", `<p class="hint">Backtest wall time: ${Object.entries(t).map(([k, v]) => `${MODEL_LABEL[k] || k} ${fmt(v, 0)}s`).join(" · ")}</p>`);

  const sl = (models.slices || []).filter((r) => r.metric === "internet");
  const groups = [...new Set(sl.map((r) => `${r.cell_tier} · ${r.day_type}`))];
  Plotly.newPlot("slices", modelsList.map((mod, i) => ({
    type: "bar", name: MODEL_LABEL[mod] || mod, x: groups,
    y: groups.map((g) => sl.find((r) => r.model === mod && `${r.cell_tier} · ${r.day_type}` === g)?.mase),
    marker: { color: palette()[i] },
  })), layout({ barmode: "group", yaxis: { ...layout().yaxis, title: { text: "MASE" } } }), CONFIG);
}

/* ------------------------------------------------------------------ capacity */
function renderCapacity() {
  const rows = state.data.capacity.cells;
  const top = rows.slice(0, 40);
  const color = (s) => ({ over_capacity: css("--bad"), at_risk: css("--warn"), watch: css("--accent"), ok: css("--ok") }[s]);
  Plotly.newPlot("cap-chart", [{
    type: "bar", x: top.map((r) => String(r.cell_id)), y: top.map((r) => r.utilisation),
    error_y: { type: "data", symmetric: false, array: top.map((r) => r.utilisation_upper - r.utilisation), arrayminus: top.map(() => 0), color: css("--muted") },
    marker: { color: top.map((r) => color(r.status)) }, hovertemplate: "cell %{x}<br>utilisation %{y:.0%}<extra></extra>",
  }], layout({
    xaxis: { ...layout().xaxis, type: "category", title: { text: "cell" } },
    yaxis: { ...layout().yaxis, tickformat: ".0%" },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: css("--bad"), dash: "dash" } }],
  }), CONFIG);
  table($("#cap-table"), [
    { label: "Cell", key: "cell_id", num: true },
    { label: "Status", render: (r) => `<span class="st-${r.status}">${r.status.replace("_", " ")}</span>` },
    { label: "Peak (UTC)", render: (r) => shortTs(r.peak_ts) },
    { label: "Peak forecast", render: (r) => fmtInt(r.peak_forecast), num: true },
    { label: "Upper bound", render: (r) => fmtInt(r.max_upper), num: true },
    { label: "Capacity", render: (r) => fmtInt(r.capacity), num: true },
    { label: "Utilisation", render: (r) => pct(r.utilisation, 0), num: true },
    { label: "Source", key: "capacity_source" },
  ], rows.filter((r) => r.status !== "ok").slice(0, 300), (r) => openCell(r.cell_id, "internet"));
}

/* ------------------------------------------------------------------ monitoring */
function renderMonitoring() {
  const mon = state.data.monitoring;
  const h = mon.data_health, load = mon.alert_load, rt = mon.retrain;
  $("#mon-kpis").innerHTML = [
    ["Retrain needed", rt.retrain ? "yes" : "no"],
    ["Data freshness", `${fmt(h.freshness_hours, 1)} h`],
    ["Missing-hour rate", pct(h.missing_hour_rate, 3)],
    ["Incidents / shift", fmt(load.incidents_per_shift, 2)],
    ["Degrading cells", fmtInt(mon.degrading_cells)],
    ["Operator precision", mon.operator_feedback ? pct(mon.operator_feedback.precision) : "no feedback yet"],
  ].map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  const d = mon.error_drift;
  const metrics = [...new Set(d.map((r) => r.metric))];
  Plotly.newPlot("drift", metrics.flatMap((m, i) => {
    const r = d.filter((x) => x.metric === m && x.fold > 0);
    return [
      { x: r.map((x) => x.cutoff.slice(0, 10)), y: r.map((x) => x.mase_model), name: m, mode: "lines+markers", line: { color: palette()[i] } },
      { x: r.map((x) => x.cutoff.slice(0, 10)), y: r.map((x) => x.mase_baseline), name: `${m} naive`, mode: "lines", line: { color: palette()[i], dash: "dot", width: 1 }, showlegend: false },
    ];
  }), layout({ yaxis: { ...layout().yaxis, title: { text: "MASE (dotted = seasonal naive)" } } }), CONFIG);

  const rowsH = [
    ["Last observation", shortTs(h.last_observation)],
    ["Input rows", fmtInt(h.input_rows)],
    ["Duplicate rows removed", fmtInt(h.duplicate_rows)],
    ["Negative values nulled", fmtInt(h.negative_values)],
    ["Catch-up dumps redistributed", fmtInt(h.catchup_dumps_fixed)],
    ["Slots interpolated (≤30 min gaps)", fmtInt(h.interpolated_slots)],
    ["Slots left missing (long gaps)", fmtInt(h.long_gap_slots)],
    ["Retrain reasons", rt.reasons.length ? rt.reasons.join("; ") : "none"],
  ];
  $("#health").innerHTML = `<div class="table-wrap"><table><tbody>${rowsH.map(([k, v]) => `<tr><td>${k}</td><td class="num">${esc(v)}</td></tr>`).join("")}</tbody></table></div>`;
}

document.addEventListener("DOMContentLoaded", () => {
  if (window.Plotly) boot();
  else window.addEventListener("load", boot);
});
