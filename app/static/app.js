/* 当前工作区与后台任务统一同步；刷新页面后从服务端恢复。 */
const $ = (id) => document.getElementById(id);
const kindNames = { import: '数据导入', scramble: '图像置乱', evaluate: 'AI 评估', select: '切换数据集' };
let state = null, submitting = null, refreshing = false, connected = false;
let fingerprint = null, clientError = '', metricRequest = 0;
let stopping = false, uploadController = null;
let savedModels = [], evaluationDatasets = [], shownEvaluationJob = null;

function updateEvaluationOptions(resetModel = false) {
  const saved = $('evaluation-mode').value === 'saved';
  $('saved-options').hidden = !saved;
  $('train-options').hidden = saved;
  $('btn-evaluate').textContent = saved ? '开始模型测试' : '开始训练并评估';
  const model = savedModels.find(m => m.id === $('saved-model').value);
  if (resetModel && model) {
    if (evaluationDatasets.some(d => d.id === model.dataset_id)) $('evaluation-dataset').value = model.dataset_id;
    document.querySelectorAll('#evaluation-iterations input').forEach(input => {
      input.checked = Number(input.value) === (model.trained_iterations || 2);
    });
  }
  $('saved-model-info').textContent = model ?
    `${model.arch} · 输入 ${model.image_size}×${model.image_size} · 训练输入：${model.trained_iterations ? `Arnold ${model.trained_iterations} 次` : '原图'} · 原测试成绩 ${model.historical_accuracy == null ? '未记录' : (model.historical_accuracy * 100).toFixed(2) + '%'}（仅对应原实验） · ${model.experiment_id}` :
    '暂无可用模型，请切换至“重新训练并评估”生成模型。';
  const scope = $('evaluation-conditions').value;
  $('saved-input-options').hidden = scope === 'original';
  const dataset = evaluationDatasets.find(d => d.id === $('evaluation-dataset').value);
  const ks = [...document.querySelectorAll('#evaluation-iterations input:checked')].map(c => Number(c.value));
  const tests = (scope !== 'scrambled' ? 1 : 0) + (scope !== 'original' ? ks.length : 0);
  const source = $('evaluation-input').value === 'saved' ? '读取已保存的置乱图，包含保存图片时的编码影响。' :
    '原图按模型尺寸处理后临时置乱，不经过 JPEG 再压缩，不生成图片文件。';
  $('evaluation-preview').textContent = model && dataset ?
    `本次使用 ${dataset.name} 的 ${Object.values(dataset.counts.test).reduce((a, b) => a + b, 0)} 张测试图，评估 ${tests} 组输入。${scope === 'original' ? '' : source}模型权重保持不变，结果独立保存。` : '请选择模型和测试数据集。';
}

async function api(path, options) {
  const res = await fetch('/api' + path, { cache: 'no-store', ...options });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const detail = (await res.json()).detail;
      msg = typeof detail === 'string' ? detail : '参数不符合要求，请检查输入';
    } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}
function setStatus(id, text, ok = true) {
  $(id).textContent = text;
  $(id).className = 'status ' + (ok ? 'ok' : 'err');
}
function setBar(id, percent) {
  $(id).style.width = Math.max(0, Math.min(100, percent)) + '%';
}
function lockControls() {
  const busy = Boolean(submitting || state?.busy || !connected);
  document.querySelectorAll('[data-mutates]').forEach(el => { el.disabled = busy; });
  $('btn-scramble').disabled = busy || !state?.imported;
  const saved = $('evaluation-mode').value === 'saved';
  $('btn-evaluate').disabled = busy || (saved ? !$('saved-model').value || !$('evaluation-dataset').value : !state?.imported);
  $('btn-select-dataset').disabled = busy || !$('dataset-select').value ||
    $('dataset-select').value === state?.dataset?.id;
  const active = state?.active_job;
  const waiting = stopping || active?.status === 'stopping';
  $('btn-stop-task').disabled = !connected || waiting ||
    !(active?.can_stop || (!active && submitting === 'import' && uploadController));
  $('btn-stop-task').textContent = waiting ? '正在停止…' : '停止当前任务';
}
function displayState() {
  lockControls();
  if (!state) return;
  const active = state.active_job;
  $('training-profile').textContent = active?.params?.training_mode === 'saved_model_evaluation' ?
    '正在使用已保存模型测试，模型参数不更新；停止后会清理本次新生成的评估结果。' :
    active?.params?.training_mode === 'transfer_linear_probe' ?
    '正在使用预训练特征自动调参；普通训练的轮数和学习率不参与本次训练。' :
    active?.params?.training_mode === 'frozen_model_scramble' ?
    '正在加载已保存的原图模型直接识别同一测试集的置乱图像，不重新训练。' :
    active?.params?.training_mode === 'adaptive_scramble_finetune' ?
    (active.params.augmentation ? '训练原图先增强、再置乱；验证和测试保持固定。' : '') +
    '正在用 Arnold 置乱图片微调全部网络层，按验证集表现选择模型，测试集在训练结束后评估。' : '';
  const pendingKind = active?.kind || submitting;
  if (active?.kind === 'evaluate' && active.params.training_mode !== 'saved_model_evaluation') {
    for (const [field, key] of [['arch', 'arch'], ['epochs', 'epochs'], ['batch-size', 'batch_size'], ['lr', 'lr'], ['image-size', 'image_size']]) {
      if (active.params[key] != null) $(field).value = active.params[key];
    }
  }
  const changing = ['import', 'scramble'].includes(pendingKind);
  setStatus('import-status', state.imported && pendingKind !== 'import' ? '已导入 ✓' : '未导入', Boolean(state.imported));
  setStatus('scramble-status', !changing && state.scrambled.length ? `已置乱：${state.scrambled.join(', ')} 次` : '未置乱', !changing && state.scrambled.length > 0);
  setStatus('evaluate-status', !changing && state.evaluated.length ? `已评估：${state.evaluated.length} 项` : '未评估', !changing && state.evaluated.length > 0);
  setBar('import-bar', state.imported && pendingKind !== 'import' ? 100 : 0);
  setBar('scramble-bar', !changing && state.scrambled.length ? 100 : 0);
  setBar('evaluate-bar', !changing && state.evaluated.length ? 100 : 0);
  const job = active || state.latest_job;
  if (!active && job?.status === 'done' && job.params.training_mode === 'saved_model_evaluation') {
    setStatus('evaluate-status', `上次模型测试已完成：${job.result?.summary?.length || 0} 组`);
    setBar('evaluate-bar', 100);
  }
  const prefix = {import: 'import', scramble: 'scramble', evaluate: 'evaluate'}[job?.kind];
  if (prefix && (active || ['error', 'cancelled'].includes(job.status))) {
    setStatus(prefix + '-status', job.status === 'error' ? job.error : job.message, job.status !== 'error');
    setBar(prefix + '-bar', job.progress);
  }
  if (submitting && !active) {
    const id = { import: 'import', scramble: 'scramble', evaluate: 'evaluate' }[submitting];
    if (id) setStatus(id + '-status', submitting === 'import' ? '正在上传数据集…' : '正在提交任务…');
  }
  const notice = $('task-notice');
  notice.className = 'task-notice wide';
  if (clientError) {
    notice.textContent = clientError; notice.classList.add('error');
  } else if (active?.status === 'stopping') {
    notice.textContent = `${active.message} · ${active.progress.toFixed(1)}%。本次任务生成的文件将自动清理，清理完成后可以继续操作。`;
  } else if (active) {
    notice.textContent = `${kindNames[active.kind] || active.kind}进行中 · ${active.progress.toFixed(1)}% · ${active.message}。完成后可执行下一步，刷新页面可恢复进度。`;
  } else if (submitting) {
    notice.textContent = '正在提交任务，请稍候…';
  } else if (job?.status === 'error') {
    notice.textContent = `${kindNames[job.kind] || job.kind}失败：${job.error}`; notice.classList.add('error');
  } else if (job?.status === 'cancelled') {
    const cleanup = job.cleanup_complete === true ? '本次任务生成的文件已自动清理，之前完成的数据和模型保留。' :
      job.cleanup_complete === false ? '部分文件清理失败：' + job.cleanup_errors.join('；') : '该历史任务停止时保留了文件。';
    notice.textContent = `${kindNames[job.kind] || job.kind}已停止 · ${job.progress.toFixed(1)}%。${cleanup}可以继续操作。`;
    if (job.cleanup_complete === false) notice.classList.add('error');
  } else {
    notice.textContent = state.dataset ? '当前无运行任务。数据集和实验均按版本保存，可在下方切换查看。' : '请先导入数据集，历史实验会自动保留。';
  }
  const d = state.dataset;
  if (d) {
    const n = d.counts;
    const validation = n.val ? ` / 验证 ${n.val.cat + n.val.dog} 张` : '';
    $('dataset-info').textContent = `${d.name} · 训练 ${n.train.cat + n.train.dog} 张${validation} / 测试 ${n.test.cat + n.test.dog} 张 · 版本 ${d.id}`;
  } else $('dataset-info').textContent = '暂无数据集';
}
async function refreshHistory() {
  const [catalog, history, models] = await Promise.all([api('/datasets'), api('/experiments'), api('/models')]);
  const modelSelect = $('saved-model'), previousModel = modelSelect.value;
  savedModels = models.models;
  modelSelect.replaceChildren();
  for (const m of savedModels) modelSelect.add(new Option(`${m.name} · ${m.experiment_id}`, m.id));
  if (!savedModels.length) modelSelect.add(new Option('暂无可用模型', ''));
  if (savedModels.some(m => m.id === previousModel)) modelSelect.value = previousModel;
  const target = $('evaluation-dataset'), previousDataset = target.value;
  evaluationDatasets = catalog.datasets.filter(d => d.status === 'ready');
  target.replaceChildren();
  for (const d of evaluationDatasets) target.add(new Option(`${d.name} · ${Object.values(d.counts.test).reduce((a, b) => a + b, 0)} 张测试图 · ${d.id}`, d.id));
  if (!evaluationDatasets.length) target.add(new Option('暂无测试数据集', ''));
  if (evaluationDatasets.some(d => d.id === previousDataset)) target.value = previousDataset;
  updateEvaluationOptions(!previousModel || !savedModels.some(m => m.id === previousModel));
  const ds = $('dataset-select');
  ds.replaceChildren();
  for (const d of catalog.datasets) {
    const count = Object.values(d.counts).flatMap(Object.values).reduce((a,b) => a+b, 0);
    ds.add(new Option(`${d.name} · ${count} 张 · ${d.id}`, d.id));
  }
  if (!catalog.datasets.length) ds.add(new Option('暂无数据集', ''));
  ds.value = catalog.active_dataset || '';
  const exp = $('experiment-select'), selected = exp.value;
  exp.replaceChildren(new Option('当前数据集最新结果', ''));
  const names = {done: '已完成', error: '失败／中断', cancelled: '已停止', running: '运行中'};
  for (const e of history.experiments) {
    exp.add(new Option(`${new Date(e.created_at * 1000).toLocaleString()} · ${e.dataset_name} · ${e.parameters.arch || '旧版'}${e.parameters.pretrained ? ' 预训练' : ''} · ${names[e.status] || e.status} · ${e.id}`, e.id));
  }
  if (history.experiments.some(e => e.id === selected)) exp.value = selected;
  lockControls();
}
async function refreshMetrics() {
  const request = ++metricRequest, selected = $('experiment-select').value;
  try {
    const data = await api('/metrics' + (selected ? '?experiment_id=' + encodeURIComponent(selected) : ''));
    if (request !== metricRequest) return;
    renderChart(data.rows); renderTable(data.rows); renderRanking(data.ranking);
    setStatus('metrics-status', data.rows.length ? `共 ${data.rows.length} 项结果` : '暂无结果', data.rows.length > 0);
    const e = data.experiment;
    const parameters = e?.parameters;
    const settings = parameters?.training_mode === 'saved_model_evaluation' ?
      ` · ${parameters.model_name} / 仅测试，不更新权重 / test 集 / ${parameters.conditions === 'original' ? '原图输入' : parameters.input_source === 'saved' ? '读取保存的置乱图' : '临时无损置乱'}${parameters.data_audit?.warnings?.length ? ' / ' + parameters.data_audit.warnings.join('；') : ''}` :
      parameters?.training_mode === 'adaptive_scramble_finetune' ?
      ` · ${parameters.arch} 全部层微调 / Arnold ${parameters.iterations.join(', ')} 次 / ${parameters.augmentation === 'pre_scramble_light_v1' ? '训练原图先增强后置乱 / ' : ''}${e.best_epoch ? `最佳验证 ${(e.best_validation_accuracy * 100).toFixed(2)}%（第 ${e.best_epoch} 轮）` : '等待首轮验证'} / 训练 ${e.completed_epochs || 0} 轮 / 测试结果为适应后的识别率` : parameters?.training_mode === 'frozen_model_scramble' ?
      ` · 已保存 ${parameters.arch} 固定模型 / 同一测试集直接识别 / 无重新训练 / 输入 ${parameters.image_size}×${parameters.image_size}` : parameters?.training_mode === 'transfer_linear_probe' ?
      ` · ${parameters.arch} 预训练特征 / 分类器自动调参 / 输入 224×224 / 独立测试` :
      parameters?.arch ? ` · ${parameters.arch} / ${parameters.epochs} 轮 / 批大小 ${parameters.batch_size} / 学习率 ${parameters.lr} / 输入 ${parameters.image_size || 224}×${parameters.image_size || 224}${parameters.selection_split === 'val' ? ' / 验证集选模型，测试集独立评估' : ''}` : '';
    $('experiment-info').textContent = e ? `数据集：${e.dataset_name} · 实验：${e.id}${settings} · ${e.status === 'done' ? '已完成' : e.status === 'cancelled' ? '已停止，以下为已保存的部分结果' : e.status === 'error' ? '失败／中断，以下可能为部分结果' : '进行中，以下为部分结果'}` : '当前数据集尚无已完成实验，可选择历史实验查看。';
    for (const [id, kind] of [['download-evaluation-summary', 'summary'], ['download-evaluation-predictions', 'predictions']]) {
      const link = $(id);
      link.hidden = !(e?.status === 'done' && parameters?.training_mode === 'saved_model_evaluation');
      if (!link.hidden) link.href = '/api/evaluations/' + encodeURIComponent(e.id) + '/download/' + kind;
      else link.removeAttribute('href');
    }
  } catch (e) { setStatus('metrics-status', '读取失败：' + e.message, false); }
}
async function refreshState() {
  while (refreshing) await new Promise(resolve => setTimeout(resolve, 40));
  refreshing = true;
  try {
    state = await api('/status'); connected = true; displayState();
    const key = JSON.stringify([state.dataset?.id, state.dataset?.scramble_id, state.dataset?.active_experiment,
      state.active_job?.id, state.active_job?.status, state.latest_job?.id, state.latest_job?.status]);
    if (key !== fingerprint) {
      await refreshHistory();
      const completed = state.latest_job;
      if (completed?.status === 'done' && completed.params.training_mode === 'saved_model_evaluation' &&
          completed.id !== shownEvaluationJob && completed.result?.experiment_id) {
        $('experiment-select').value = completed.result.experiment_id;
        shownEvaluationJob = completed.id;
      }
      await refreshMetrics(); fingerprint = key;
    }
  } catch (e) {
    connected = false; lockControls();
    $('task-notice').textContent = '连接中断，正在重试；后台任务可能仍在执行，请勿重复提交。';
    $('task-notice').className = 'task-notice wide error';
  } finally { refreshing = false; }
}
async function submitAction(kind, path, options) {
  if (submitting || state?.busy || !connected) return;
  submitting = kind; clientError = ''; displayState();
  if (['import', 'scramble'].includes(kind) && !$('experiment-select').value) {
    ++metricRequest; renderChart([]); renderTable([]); renderRanking([]);
    setStatus('metrics-status', '等待当前任务完成');
  }
  try { await api(path, options); }
  catch (e) { if (e.name !== 'AbortError') clientError = e.message; }
  finally {
    if (kind === 'import') uploadController = null;
    submitting = null; fingerprint = null; await refreshState(); displayState();
  }
}
$('btn-import').addEventListener('click', () => {
  const file = $('import-file').files[0], value = $('max-per-class').value;
  if (!file) { setStatus('import-status', '请先选择 ZIP 文件', false); return; }
  if (value && (!Number.isInteger(Number(value)) || Number(value) < 3)) {
    setStatus('import-status', '每类至少选择 3 张图片，或留空使用全量', false); return;
  }
  if (submitting || state?.busy || !connected) return;
  const query = new URLSearchParams({filename: file.name});
  if (value) query.set('max_per_class', value);
  uploadController = new AbortController();
  submitAction('import', '/dataset/upload?' + query, {
    method: 'POST', headers: {'Content-Type': 'application/zip'},
    body: file, signal: uploadController.signal
  });
});
$('btn-stop-task').addEventListener('click', async () => {
  if (stopping) return;
  stopping = true; clientError = ''; lockControls();
  const upload = uploadController;
  const jobId = state?.active_job?.id;
  try {
    await refreshState();
    const active = state?.active_job;
    if ((!jobId || active?.id === jobId) && (active?.can_stop || active?.status === 'stopping')) {
      await api('/jobs/' + encodeURIComponent(active.id) + '/stop', {method: 'POST'});
    }
    upload?.abort();
  } catch (e) {
    await refreshState();
    if (state?.active_job) clientError = '停止失败：' + e.message;
  } finally {
    stopping = false; fingerprint = null;
    await refreshState(); displayState();
  }
});
function post(kind, path, body) {
  return submitAction(kind, path, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)});
}
$('btn-scramble').addEventListener('click', () => {
  const iterations = [...document.querySelectorAll('#iterations input:checked')].map(c => Number(c.value));
  if (!iterations.length) { setStatus('scramble-status', '请至少勾选一个迭代次数', false); return; }
  post('scramble', '/scramble', {algorithm: $('algo').value, iterations});
});
$('btn-evaluate').addEventListener('click', () => {
  if ($('evaluation-mode').value === 'saved') {
    const iterations = [...document.querySelectorAll('#evaluation-iterations input:checked')].map(c => Number(c.value));
    const conditions = $('evaluation-conditions').value, batch_size = Number($('evaluation-batch-size').value);
    if (!$('saved-model').value || !$('evaluation-dataset').value ||
        (conditions !== 'original' && !iterations.length) || !Number.isInteger(batch_size) || batch_size < 1 || batch_size > 128) {
      setStatus('evaluate-status', '请选择模型、测试数据集、置乱次数及有效批大小（1–128）', false); return;
    }
    $('experiment-select').value = '';
    post('evaluate', '/evaluate/saved', {model_id: $('saved-model').value,
      dataset_id: $('evaluation-dataset').value, conditions, iterations: conditions === 'original' ? [] : iterations,
      input_source: $('evaluation-input').value, batch_size});
    return;
  }
  const epochs = Number($('epochs').value), batch_size = Number($('batch-size').value), lr = Number($('lr').value), image_size = Number($('image-size').value);
  if (!Number.isInteger(epochs) || epochs < 1 || !Number.isInteger(batch_size) || batch_size < 1 || !Number.isFinite(lr) || lr <= 0 || !Number.isInteger(image_size) || image_size < 32 || image_size > 512) {
    setStatus('evaluate-status', '请检查轮数、批大小、学习率及输入尺寸（32–512）', false); return;
  }
  post('evaluate', '/evaluate', {arch: $('arch').value, epochs, batch_size, lr, image_size, optimizer:'adam'});
});
$('evaluation-mode').addEventListener('change', () => { updateEvaluationOptions(); lockControls(); });
$('saved-model').addEventListener('change', () => { updateEvaluationOptions(true); lockControls(); });
for (const id of ['evaluation-dataset', 'evaluation-conditions', 'evaluation-input', 'evaluation-iterations']) {
  $(id).addEventListener('change', () => { updateEvaluationOptions(); lockControls(); });
}
$('dataset-select').addEventListener('change', lockControls);
$('btn-select-dataset').addEventListener('click', () => {
  $('experiment-select').value = '';
  post('select', '/datasets/select', {dataset_id: $('dataset-select').value});
});
$('experiment-select').addEventListener('change', refreshMetrics);
$('btn-metrics').addEventListener('click', async () => { await refreshHistory(); await refreshMetrics(); });
async function init() {
  lockControls();
  try {
    const cfg = await api('/config');
    for (const k of cfg.iterations) {
      const label = document.createElement('label'); label.className = 'chip';
      label.innerHTML = `<input type="checkbox" value="${k}" data-mutates checked> ${k} 次`;
      $('iterations').appendChild(label);
      const evaluation = document.createElement('label'); evaluation.className = 'chip';
      evaluation.innerHTML = `<input type="checkbox" value="${k}" data-mutates ${k === 2 ? 'checked' : ''}> ${k} 次`;
      $('evaluation-iterations').appendChild(evaluation);
    }
    $('epochs').value = cfg.defaults.epochs; $('batch-size').value = cfg.defaults.batch_size; $('lr').value = cfg.defaults.lr; $('image-size').value = cfg.image_size;
  } catch (_) {}
  const tick = async () => { await refreshState(); setTimeout(tick, 1200); };
  await tick();
}

function interpret(acc) {
  if (acc == null) return '—';
  const d = Math.abs(acc - 0.5);
  if (d < 0.05) return '接近随机猜测基线；不能单独证明安全';
  if (d < 0.15) return '偏离随机基线，仍有可识别信息';
  return '明显偏离随机基线';
}

function renderTable(rows) {
  const tb = $('metrics-table').querySelector('tbody');
  tb.innerHTML = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    const label = r.iterations == null ? '原始图（未置乱）' : `Arnold ${r.iterations} 次`;
    const percent = value => value == null ? '—' : (value * 100).toFixed(2) + '%';
    tr.innerHTML = `<td>${label}</td><td>${percent(r.test_acc)}</td><td>${percent(r.precision)}</td><td>${percent(r.recall)}</td><td>${percent(r.f1)}</td><td>${r.total ?? '—'}</td><td>${r.iterations == null ? '原图输入表现' : interpret(r.test_acc)}</td>`;
    tb.appendChild(tr);
  });
  const matrices = $('confusion-matrices'); matrices.replaceChildren();
  for (const row of rows.filter(r => r.confusion_matrix && r.classes)) {
    const table = document.createElement('table');
    const caption = table.createCaption();
    caption.textContent = row.iterations == null ? '原图' : `Arnold ${row.iterations} 次`;
    const header = table.insertRow();
    for (const text of ['真实／预测', ...row.classes]) {
      const th = document.createElement('th'); th.textContent = text; header.appendChild(th);
    }
    row.confusion_matrix.forEach((counts, index) => {
      const tr = table.insertRow();
      for (const text of [row.classes[index], ...counts]) tr.insertCell().textContent = text;
    });
    matrices.appendChild(table);
  }
  $('confusion-details').hidden = !matrices.children.length;
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
      <td>${(r.test_acc * 100).toFixed(2)}%</td><td>${Math.abs(r.test_acc - 0.5).toFixed(4)}</td>`;
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
      ${(r.test_acc * 100).toFixed(2)}%</text>`;
  });

  svg.innerHTML = svgInner;
}

init();
