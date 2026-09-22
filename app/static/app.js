/* 图像置乱算法 AI 安全评价系统 —— 前端逻辑（无第三方依赖，SVG 手绘图表） */

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const res = await fetch('/api' + path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ---------- 任务轮询 ---------- */
function pollJob(jobId, { onProgress, onDone, onError }) {
  const timer = setInterval(async () => {
    try {
      const job = await api('/jobs/' + jobId);
      onProgress && onProgress(job);
      if (job.status === 'done') {
        clearInterval(timer);
        onDone && onDone(job);
      } else if (job.status === 'error') {
        clearInterval(timer);
        onError && onError(job);
      }
    } catch (e) {
      clearInterval(timer);
      onError && onError({ error: e.message });
    }
  }, 800);
}

function setStatus(id, text, ok = true) {
  const el = $(id);
  el.textContent = text;
  el.className = 'status ' + (ok ? 'ok' : 'err');
}

function setBar(id, percent) {
  $(id).style.width = Math.max(0, Math.min(100, percent)) + '%';
}

/* ---------- 初始化：加载配置与状态 ---------- */
async function init() {
  try {
    const cfg = await api('/config');
    const chips = $('iterations');
    cfg.iterations.forEach((k) => {
      const label = document.createElement('label');
      label.className = 'chip';
      label.innerHTML = `<input type="checkbox" value="${k}" checked> ${k} 次`;
      chips.appendChild(label);
    });
    $('epochs').value = cfg.defaults.epochs;
    $('batch-size').value = cfg.defaults.batch_size;
    $('lr').value = cfg.defaults.lr;
  } catch (e) {
    setStatus('import-status', '无法连接后端：' + e.message, false);
  }
  await refreshStatus();
  await refreshMetrics();
}

async function refreshStatus() {
  try {
    const s = await api('/status');
    setStatus('import-status', s.imported ? '已导入 ✓' : '未导入', s.imported);
    setStatus('scramble-status',
      s.scrambled.length ? `已置乱：${s.scrambled.join(', ')} 次` : '未置乱', s.scrambled.length > 0);
    setStatus('evaluate-status',
      s.evaluated.length ? `已评估：${s.evaluated.length} 项` : '未评估', s.evaluated.length > 0);
  } catch (e) {
    setStatus('import-status', '状态读取失败：' + e.message, false);
  }
}

/* ---------- 模块① 导入 ---------- */
$('btn-import').addEventListener('click', async () => {
  const file = $('import-file').files[0];
  if (!file) { setStatus('import-status', '请先选择 zip 文件', false); return; }
  const fd = new FormData();
  fd.append('file', file);
  const mp = $('max-per-class').value;
  if (mp) fd.append('max_per_class', mp);

  setStatus('import-status', '上传中…');
  setBar('import-bar', 5);
  try {
    const { job_id } = await api('/dataset/import', { method: 'POST', body: fd });
    pollJob(job_id, {
      onProgress: (j) => { setBar('import-bar', j.progress); setStatus('import-status', j.message); },
      onDone: async () => {
        setBar('import-bar', 100);
        setStatus('import-status', '导入完成 ✓', true);
        await refreshStatus();
      },
      onError: (j) => { setStatus('import-status', '导入失败：' + (j.error || j.message), false); },
    });
  } catch (e) {
    setStatus('import-status', '上传失败：' + e.message, false);
  }
});

/* ---------- 模块② 置乱 ---------- */
$('btn-scramble').addEventListener('click', async () => {
  const iterations = [...document.querySelectorAll('#iterations input:checked')]
    .map((c) => parseInt(c.value));
  if (!iterations.length) { setStatus('scramble-status', '请至少勾选一个迭代次数', false); return; }
  const algorithm = $('algo').value;

  setStatus('scramble-status', '已提交置乱任务…');
  setBar('scramble-bar', 0);
  try {
    const { job_id } = await api('/scramble', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algorithm, iterations }),
    });
    pollJob(job_id, {
      onProgress: (j) => { setBar('scramble-bar', j.progress); setStatus('scramble-status', j.message); },
      onDone: async () => {
        setBar('scramble-bar', 100);
        setStatus('scramble-status', '置乱完成 ✓', true);
        await refreshStatus();
      },
      onError: (j) => { setStatus('scramble-status', '置乱失败：' + (j.error || j.message), false); },
    });
  } catch (e) {
    setStatus('scramble-status', '提交失败：' + e.message, false);
  }
});

/* ---------- 模块③ 评估 ---------- */
$('btn-evaluate').addEventListener('click', async () => {
  const body = {
    arch: $('arch').value,
    epochs: parseInt($('epochs').value) || 3,
    batch_size: parseInt($('batch-size').value) || 16,
    lr: parseFloat($('lr').value) || 0.001,
    optimizer: 'adam',
  };
  setStatus('evaluate-status', '已提交评估任务…');
  setBar('evaluate-bar', 0);
  try {
    const { job_id } = await api('/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    pollJob(job_id, {
      onProgress: (j) => { setBar('evaluate-bar', j.progress); setStatus('evaluate-status', j.message); },
      onDone: async () => {
        setBar('evaluate-bar', 100);
        setStatus('evaluate-status', '评估完成 ✓', true);
        await refreshStatus();
        await refreshMetrics();
      },
      onError: (j) => { setStatus('evaluate-status', '评估失败：' + (j.error || j.message), false); },
    });
  } catch (e) {
    setStatus('evaluate-status', '提交失败：' + e.message, false);
  }
});

/* ---------- 模块④ 指标 ---------- */
$('btn-metrics').addEventListener('click', refreshMetrics);

async function refreshMetrics() {
  try {
    const data = await api('/metrics');
    renderChart(data.rows);
    renderTable(data.rows);
    renderRanking(data.ranking);
    setStatus('metrics-status', data.rows.length ? `共 ${data.rows.length} 项结果` : '暂无结果', data.rows.length > 0);
  } catch (e) {
    setStatus('metrics-status', '读取失败：' + e.message, false);
  }
}

function interpret(acc) {
  if (acc == null) return '—';
  const d = Math.abs(acc - 0.5);
  if (d < 0.05) return '接近 50%，安全性高 ✓';
  if (d < 0.15) return '有一定残留，安全性中等';
  return '残留明显，安全性低';
}

function renderTable(rows) {
  const tb = $('metrics-table').querySelector('tbody');
  tb.innerHTML = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    const label = r.iterations == null ? '原始图（未置乱）' : `Arnold ${r.iterations} 次`;
    tr.innerHTML = `<td>${label}</td><td>${(r.test_acc * 100).toFixed(1)}%</td><td>${interpret(r.test_acc)}</td>`;
    tb.appendChild(tr);
  });
}

function renderRanking(ranking) {
  const tb = $('ranking-table').querySelector('tbody');
  tb.innerHTML = '';
  if (!ranking.length) {
    tb.innerHTML = '<tr><td colspan="4">暂无置乱评估结果</td></tr>';
    return;
  }
  ranking.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td><td>${r.iterations} 次</td>
      <td>${(r.test_acc * 100).toFixed(1)}%</td><td>${Math.abs(r.test_acc - 0.5).toFixed(3)}</td>`;
    tb.appendChild(tr);
  });
}

/* SVG 手绘曲线 */
function renderChart(rows) {
  const svg = $('chart');
  if (!rows.length) { svg.innerHTML = ''; return; }

  const W = 640, H = 360, padL = 56, padR = 20, padT = 20, padB = 46;
  const innerW = W - padL - padR, innerH = H - padT - padB;

  const labels = rows.map((r) => (r.iterations == null ? '原始' : String(r.iterations)));
  const xs = labels.map((_, i) => padL + (i / Math.max(1, labels.length - 1)) * innerW);
  const yOf = (v) => padT + (1 - v) * innerH;   // v ∈ [0,1]

  let svgInner = '';
  // 网格 + y 轴刻度
  for (let g = 0; g <= 4; g++) {
    const v = g / 4, y = yOf(v);
    svgInner += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#eee"/>`;
    svgInner += `<text x="${padL - 8}" y="${y + 4}" text-anchor="end" class="axis">${(v * 100).toFixed(0)}%</text>`;
  }
  // 50% 基线
  svgInner += `<line x1="${padL}" y1="${yOf(0.5)}" x2="${W - padR}" y2="${yOf(0.5)}"
    stroke="#e74c3c" stroke-dasharray="6 4"/>
    <text x="${W - padR}" y="${yOf(0.5) - 6}" text-anchor="end" fill="#e74c3c" class="axis">50% 基线</text>`;
  // x 轴标签
  labels.forEach((lb, i) => {
    svgInner += `<text x="${xs[i]}" y="${H - padB + 18}" text-anchor="middle" class="axis">${lb}</text>`;
  });

  // 折线 + 数据点
  const pts = rows.map((r, i) => `${xs[i]},${yOf(r.test_acc)}`);
  if (pts.length > 1) {
    svgInner += `<polyline points="${pts.join(' ')}" fill="none" stroke="#3498db" stroke-width="2.5"/>`;
  }
  rows.forEach((r, i) => {
    svgInner += `<circle cx="${xs[i]}" cy="${yOf(r.test_acc)}" r="5" fill="#3498db"/>
      <text x="${xs[i]}" y="${yOf(r.test_acc) - 10}" text-anchor="middle" class="axis" fill="#2c3e50">
      ${(r.test_acc * 100).toFixed(1)}%</text>`;
  });

  svg.innerHTML = svgInner;
}

init();
