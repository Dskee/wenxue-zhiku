'use strict';

/* ------------------------------------------------------------------ *
 * 小工具
 * ------------------------------------------------------------------ */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

let toastTimer = null;
function toast(msg, isErr) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast'; }, isErr ? 5200 : 2600);
}

/** 从后端错误里拿到可读文字：后端统一用 {"detail": ...}。 */
async function readError(res) {
  try {
    const data = await res.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
    return JSON.stringify(data);
  } catch {
    const text = await res.text().catch(() => '');
    return text || ('HTTP ' + res.status);
  }
}

async function api(path, { method = 'GET', body, form, query } = {}) {
  let url = '/api' + path;
  if (query) {
    const qs = new URLSearchParams(query).toString();
    if (qs) url += '?' + qs;
  }
  const opts = { method, headers: {} };
  if (form) {
    opts.body = form; // 交给浏览器自己设置 multipart 边界
  } else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(await readError(res));
  const type = res.headers.get('content-type') || '';
  return type.includes('application/json') ? res.json() : res.text();
}

const escapeHtml = (s) =>
  String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));

const shortText = (s, n = 90) => {
  const t = String(s || '').replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n) + '…' : t;
};

/* ------------------------------------------------------------------ *
 * 全局状态
 * ------------------------------------------------------------------ */
const state = {
  kbList: [],
  sessionId: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now())),
  history: [],          // [[问, 答], ...]
  currentFile: null,
  ankiCards: [],
};

/* ------------------------------------------------------------------ *
 * 视图切换
 * ------------------------------------------------------------------ */
function switchView(name) {
  $$('.nav-item').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
  $$('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + name));
  if (name === 'kb') loadKbList();
  if (name === 'files') loadFiles();
  if (name === 'settings') loadSettings();
}

$$('.nav-item').forEach((btn) => {
  btn.addEventListener('click', () => switchView(btn.dataset.view));
});

/* ------------------------------------------------------------------ *
 * 后端状态
 * ------------------------------------------------------------------ */
async function checkBackend() {
  const el = $('#backend-status');
  try {
    const s = await api('/knowledge/status');
    const rerank = s.rerank || {};
    const tip = rerank.rerank_loaded
      ? '重排已加载'
      : rerank.rerank_error
        ? '重排不可用（降级为混合检索）'
        : '重排待首次检索加载';
    el.textContent = '后端就绪 · ' + tip;
    el.className = 'status ok';
  } catch (e) {
    el.textContent = '后端未连接：' + shortText(e.message, 60);
    el.className = 'status bad';
  }
}

/* ------------------------------------------------------------------ *
 * 知识库列表（多处下拉框共用）
 * ------------------------------------------------------------------ */
function fillKbSelects() {
  const targets = ['#chat-kb', '#upload-kb', '#files-kb', '#summary-kb', '#quiz-kb', '#anki-kb'];
  targets.forEach((sel) => {
    const el = $(sel);
    if (!el) return;
    const keep = el.value;
    const isChat = sel === '#chat-kb' || sel === '#summary-kb';
    el.innerHTML = (isChat ? '<option value="">' + (sel === '#chat-kb' ? '自动选择' : '自动匹配 / 新建') + '</option>' : '') +
      state.kbList.map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`).join('');
    if (keep && state.kbList.includes(keep)) el.value = keep;
  });
}

async function loadKbList() {
  try {
    state.kbList = await api('/knowledge/queryall_knowledge');
    fillKbSelects();
    renderKbList();
  } catch (e) {
    $('#kb-list').innerHTML = `<p class="muted">读取失败：${escapeHtml(e.message)}</p>`;
  }
}

function renderKbList() {
  const box = $('#kb-list');
  if (!state.kbList.length) {
    box.innerHTML = '<p class="muted">还没有知识库。先在上面的表单里建一个。</p>';
    return;
  }
  box.innerHTML = state.kbList.map((name) => `
    <div class="kb-item">
      <div class="meta"><strong>${escapeHtml(name)}</strong><small>点击右侧按钮管理</small></div>
      <div class="row">
        <button class="ghost" data-act="files" data-kb="${escapeHtml(name)}">看文件</button>
        <button class="danger" data-act="del" data-kb="${escapeHtml(name)}">删除</button>
      </div>
    </div>`).join('');
}

$('#kb-list').addEventListener('click', async (ev) => {
  const btn = ev.target.closest('button[data-act]');
  if (!btn) return;
  const kb = btn.dataset.kb;
  if (btn.dataset.act === 'files') {
    $('#files-kb').value = kb;
    switchView('files');
  } else if (btn.dataset.act === 'del') {
    if (!confirm(`删除知识库「${kb}」？索引与笔记文件都会被删掉，不可撤销。`)) return;
    try {
      await api('/knowledge/del_knowledge', { method: 'DELETE', query: { name: kb } });
      toast('已删除 ' + kb);
      loadKbList();
    } catch (e) { toast(e.message, true); }
  }
});

/* ------------------------------------------------------------------ *
 * 问答
 * ------------------------------------------------------------------ */
const chatBox = $('#chat-messages');
const chatInput = $('#chat-input');

function pushMessage(role, text) {
  const empty = chatBox.querySelector('.empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div.querySelector('.bubble');
}

chatInput.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    $('#chat-form').requestSubmit();
  }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
});

$('#chat-files').addEventListener('change', (ev) => {
  const names = Array.from(ev.target.files).map((f) => f.name);
  $('#chat-files-hint').textContent = names.length ? '已选 ' + names.join('、') : '';
});

$('#chat-clear').addEventListener('click', () => {
  state.history = [];
  chatBox.innerHTML = '<div class="empty"><p>问点什么吧。默认会自动挑最相关的知识库检索，也可以在右上角指定。</p></div>';
});

$('#chat-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  const kbName = $('#chat-kb').value;
  const files = $('#chat-files').files;

  chatInput.value = '';
  chatInput.style.height = 'auto';
  pushMessage('user', text);
  const bubble = pushMessage('bot', '…');
  $('#chat-send').disabled = true;

  const form = new FormData();
  $('#chat-files-hint').textContent = '';
  form.append('query', text + '\n');
  Array.from(files).forEach((f) => form.append('files', f));
  form.append('sys_prompt', $('#chat-sys').value || '');
  form.append('history_len', '-1');
  form.append('temperature', $('#chat-temp').value || '0.7');
  form.append('max_tokens', $('#chat-maxtok').value || '4096');
  form.append('session_id', state.sessionId);
  form.append('kb_name', kbName || '');
  state.history.forEach((pair) => form.append('history', JSON.stringify(pair)));

  let answer = '';
  try {
    const res = await fetch('/api/chat/chatbot', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await readError(res));

    // NDJSON：每行一个 {"answer": ...}。按行缓存，容忍任意拆包。
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line) continue;
        try {
          answer += JSON.parse(line).answer || '';
          bubble.textContent = answer;
          chatBox.scrollTop = chatBox.scrollHeight;
        } catch { /* 单行不是 JSON 就跳过，不中断整段回答 */ }
      }
    }
    const tail = buffer.trim();
    if (tail) {
      try { answer += JSON.parse(tail).answer || ''; } catch { /* ignore */ }
    }
    bubble.textContent = answer || '（没有返回内容）';
    if (answer) state.history.push([text, answer]);
  } catch (e) {
    bubble.textContent = '出错了：' + e.message;
  } finally {
    $('#chat-send').disabled = false;
    $('#chat-files').value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});

/* ------------------------------------------------------------------ *
 * 建库 / 上传
 * ------------------------------------------------------------------ */
$('#kb-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const name = $('#kb-name').value.trim();
  const description = $('#kb-desc').value.trim();
  if (!name || !description) return toast('名称和描述都要填', true);
  try {
    await api('/knowledge/add_knowledge', { method: 'POST', body: { name, description } });
    toast('已创建 ' + name);
    $('#kb-form').reset();
    loadKbList();
  } catch (e) { toast(e.message, true); }
});

$('#upload-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const kb = $('#upload-kb').value;
  const files = $('#upload-files').files;
  if (!kb) return toast('先选一个知识库', true);
  if (!files.length) return toast('先选文件', true);

  const form = new FormData();
  Array.from(files).forEach((f) => form.append('files', f));
  form.append('kb_name', kb);
  form.append('chunk_size', $('#upload-chunk').value || '128');
  form.append('chunk_overlap', $('#upload-overlap').value || '20');

  const btn = ev.target.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.textContent = '上传中…';
  try {
    const d = await api('/knowledge/upload_docs', { method: 'POST', form });
    toast('上传成功：' + shortText(JSON.stringify(d), 60));
    $('#upload-form').reset();
    fillKbSelects();
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.textContent = '上传并建索引';
  }
});

/* ------------------------------------------------------------------ *
 * 文件管理
 * ------------------------------------------------------------------ */
async function loadFiles() {
  const kb = $('#files-kb').value;
  const list = $('#file-list');
  if (!kb) {
    list.innerHTML = '<li class="muted">先选知识库</li>';
    return;
  }
  list.innerHTML = '<li class="muted">加载中…</li>';
  try {
    const d = await api('/knowledge/list_files', { method: 'POST', query: { kb_name: kb } });
    const files = d.files || [];
    if (!files.length) {
      list.innerHTML = '<li class="muted">这个库还没有文件</li>';
      return;
    }
    list.innerHTML = files.map((f) => `<li data-file="${escapeHtml(f)}">${escapeHtml(f)}</li>`).join('');
  } catch (e) {
    list.innerHTML = `<li class="muted">读取失败：${escapeHtml(e.message)}</li>`;
  }
}

$('#files-kb').addEventListener('change', () => {
  state.currentFile = null;
  $('#file-content').value = '';
  $('#file-content').disabled = true;
  $('#file-save').disabled = true;
  $('#file-delete').disabled = true;
  $('#file-current').textContent = '未选择文件';
  loadFiles();
});

$('#file-list').addEventListener('click', async (ev) => {
  const li = ev.target.closest('li[data-file]');
  if (!li) return;
  $$('#file-list li').forEach((x) => x.classList.remove('active'));
  li.classList.add('active');
  const kb = $('#files-kb').value;
  const name = li.dataset.file;
  try {
    const d = await api('/knowledge/read_file', { method: 'POST', body: { kb_name: kb, file_name: name } });
    state.currentFile = name;
    $('#file-current').textContent = kb + ' / ' + name;
    $('#file-content').value = d.content || '';
    $('#file-content').disabled = false;
    $('#file-save').disabled = false;
    $('#file-delete').disabled = false;
  } catch (e) { toast(e.message, true); }
});

$('#file-save').addEventListener('click', async () => {
  const kb = $('#files-kb').value;
  if (!kb || !state.currentFile) return;
  const btn = $('#file-save');
  btn.disabled = true;
  btn.textContent = '保存中…';
  try {
    const d = await api('/knowledge/update_file', {
      method: 'POST',
      body: { kb_name: kb, file_name: state.currentFile, content: $('#file-content').value },
    });
    toast('已保存，重建索引 ' + (d.rebuilt_chunks || 0) + ' 块');
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = '保存并重建索引'; }
});

$('#file-delete').addEventListener('click', async () => {
  const kb = $('#files-kb').value;
  if (!kb || !state.currentFile) return;
  if (!confirm(`删除文件「${state.currentFile}」？`)) return;
  try {
    await api('/knowledge/delete_file', { method: 'POST', body: { kb_name: kb, file_name: state.currentFile } });
    toast('已删除');
    state.currentFile = null;
    $('#file-content').value = '';
    $('#file-content').disabled = true;
    $('#file-save').disabled = true;
    $('#file-delete').disabled = true;
    loadFiles();
  } catch (e) { toast(e.message, true); }
});

/* ------------------------------------------------------------------ *
 * 对话整理
 * ------------------------------------------------------------------ */
$('#summary-pull').addEventListener('click', () => {
  if (!state.history.length) return toast('当前还没有对话', true);
  $('#summary-input').value = state.history.map(([q, a]) => `问：${q}\n答：${a}`).join('\n\n');
});

$('#summary-run').addEventListener('click', async () => {
  const text = $('#summary-input').value.trim();
  if (!text) return toast('先粘贴或取用对话内容', true);
  const btn = $('#summary-run');
  btn.disabled = true;
  $('#summary-hint').textContent = '整理中，可能需要十几秒…';
  try {
    const history = text.split(/\n\s*\n/).map((block) => {
      const m = block.split(/\n/);
      return { role: 'user', content: m.join('\n') };
    });
    const d = await api('/chat/summarize', { method: 'POST', body: { history, sys_prompt: '' } });
    $('#summary-output').value = d.summary || d.text || JSON.stringify(d, null, 2);
    $('#summary-hint').textContent = '已生成，检查后再保存';
  } catch (e) {
    $('#summary-hint').textContent = '';
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

$('#summary-save').addEventListener('click', async () => {
  const content = $('#summary-output').value.trim();
  if (!content) return toast('还没有可保存的内容', true);
  const kb = $('#summary-kb').value;
  try {
    const d = await api('/knowledge/save_note', {
      method: 'POST',
      body: { kb_name: kb, content, description: shortText(content, 40), file_name: '' },
    });
    toast('已保存到 ' + d.kb_name + ' / ' + d.file_name);
    loadKbList();
  } catch (e) { toast(e.message, true); }
});

/* ------------------------------------------------------------------ *
 * 出题
 * ------------------------------------------------------------------ */
$('#quiz-run').addEventListener('click', async () => {
  const kb = $('#quiz-kb').value;
  if (!kb) return toast('先选知识库', true);
  const btn = $('#quiz-run');
  btn.disabled = true;
  $('#quiz-output').textContent = '生成中，可能需要几十秒…';
  try {
    const d = await api('/knowledge/quiz', {
      method: 'POST',
      body: {
        kb_name: kb,
        question_type: $('#quiz-type').value,
        num_questions: Number($('#quiz-num').value) || 5,
        topic: $('#quiz-topic').value.trim(),
      },
    });
    $('#quiz-output').textContent = d.quiz || '';
  } catch (e) { $('#quiz-output').textContent = '失败：' + e.message; }
  finally { btn.disabled = false; }
});

/* ------------------------------------------------------------------ *
 * Anki 卡片
 * ------------------------------------------------------------------ */
$('#anki-run').addEventListener('click', async () => {
  const kb = $('#anki-kb').value;
  if (!kb) return toast('先选知识库', true);
  const btn = $('#anki-run');
  btn.disabled = true;
  $('#anki-hint').textContent = '生成中，可能需要几十秒…';
  $('#anki-cards').innerHTML = '';
  try {
    const d = await api('/knowledge/anki/preview', {
      method: 'POST',
      body: {
        kb_name: kb,
        card_type: $('#anki-type').value,
        max_cards: Number($('#anki-num').value) || 10,
        topic: $('#anki-topic').value.trim(),
      },
    });
    state.ankiCards = d.cards || [];
    $('#anki-hint').textContent = `生成 ${d.count} 张（${d.card_type_label}）` +
      (d.invalid_count ? `，${d.invalid_count} 张不符合格式已跳过` : '');
    renderAnkiCards();
    $('#anki-export').disabled = !state.ankiCards.length;
  } catch (e) {
    $('#anki-hint').textContent = '';
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

function renderAnkiCards() {
  const box = $('#anki-cards');
  if (!state.ankiCards.length) {
    box.innerHTML = '<p class="muted">还没有候选卡片。</p>';
    return;
  }
  box.innerHTML = state.ankiCards.map((c, i) => {
    const pv = c.preview || {};
    return `
      <div class="anki-card" data-i="${i}">
        <input type="checkbox" checked />
        <div class="body">
          <div class="q">${escapeHtml(pv.front || '')}</div>
          <div class="a">${escapeHtml(pv.back || pv.extra || '')}</div>
          <details><summary class="muted">编辑原始内容</summary>
            <textarea rows="4" data-raw="${i}">${escapeHtml(JSON.stringify(c.raw, null, 2))}</textarea>
          </details>
        </div>
      </div>`;
  }).join('');
}

$('#anki-all').addEventListener('click', () => {
  const boxes = $$('#anki-cards input[type="checkbox"]');
  const allOn = boxes.every((b) => b.checked);
  boxes.forEach((b) => { b.checked = !allOn; });
});

$('#anki-export').addEventListener('click', async () => {
  const kb = $('#anki-kb').value;
  const picked = [];
  $$('#anki-cards .anki-card').forEach((card) => {
    const box = card.querySelector('input[type="checkbox"]');
    if (!box.checked) return;
    const i = Number(card.dataset.i);
    const ta = card.querySelector('textarea[data-raw]');
    let raw = state.ankiCards[i].raw;
    try { raw = JSON.parse(ta.value); } catch { /* 改坏了就用原始值 */ }
    picked.push(raw);
  });
  if (!picked.length) return toast('至少选一张卡片', true);

  const btn = $('#anki-export');
  btn.disabled = true;
  try {
    const d = await api('/knowledge/anki/export', {
      method: 'POST',
      body: { kb_name: kb, card_type: $('#anki-type').value, cards: picked },
    });
    $('#anki-hint').textContent = `已导出 ${d.count} 张（跳过 ${d.skipped}），正在下载…`;
    window.location.href = '/api' + d.download;
  } catch (e) {
    toast(e.message, true);
  } finally { btn.disabled = false; }
});

/* ------------------------------------------------------------------ *
 * 设置
 * ------------------------------------------------------------------ */
let profilesLoaded = false;

async function loadProfiles() {
  if (profilesLoaded) return;
  try {
    const d = await api('/settings/profiles');
    const profiles = d.profiles || {};
    $('#set-profile').innerHTML = Object.entries(profiles)
      .map(([k, v]) => `<option value="${escapeHtml(k)}">${escapeHtml(v.label || k)}</option>`)
      .join('');
    $('#set-profile').dataset.hints = JSON.stringify(
      Object.fromEntries(Object.entries(profiles).map(([k, v]) => [k, { base: v.base_url || '', hint: v.hint || '' }]))
    );
    profilesLoaded = true;
  } catch (e) { toast('读取来源预设失败：' + e.message, true); }
}

$('#set-profile').addEventListener('change', () => {
  try {
    const hints = JSON.parse($('#set-profile').dataset.hints || '{}');
    const p = hints[$('#set-profile').value] || {};
    if (p.base) $('#set-base-url').value = p.base;
    $('#set-profile-hint').textContent = p.hint || '';
  } catch { /* ignore */ }
});

async function loadSettings() {
  await loadProfiles();
  try {
    const s = await api('/settings');
    if (s.profile) $('#set-profile').value = s.profile;
    $('#set-chat-model').value = s.chat_model || '';
    $('#set-embedding-model').value = s.embedding_model || '';
    $('#set-base-url').value = s.base_url || '';
    $('#set-embedding-base').value = s.embedding_base_url || '';
    $('#settings-output').textContent = summarizeSettings(s);
  } catch (e) {
    $('#settings-output').textContent = '读取设置失败：' + e.message;
  }
}

function summarizeSettings(s) {
  const key = s.has_key ? '已配置 ' + (s.api_key_masked || '') : (s.needs_key ? '未配置' : '不需要（本地模型）');
  return [
    '来源：' + (s.profile_label || s.profile || '(未设置)'),
    '对话模型：' + (s.chat_model || '(未设置)'),
    '向量模型：' + (s.embedding_model || '(未设置)') + (s.embedding_uses_own_config ? '（独立地址）' : ''),
    'API Key：' + key,
  ].join('\n');
}

async function fetchModels(kind) {
  const target = kind === 'chat' ? '#chat-models' : '#embedding-models';
  const current = kind === 'chat' ? $('#set-chat-model') : $('#set-embedding-model');
  try {
    const d = await api('/settings/models', { query: { kind } });
    const list = (kind === 'chat' ? d.chat_models : d.embedding_models) || [];
    $(target).innerHTML = list.map((m) => `<option value="${escapeHtml(m)}"></option>`).join('');
    if (!current.value && list.length) current.value = list[0];
    toast(d.source === 'provider'
      ? `拉取到 ${d.total || list.length} 个模型`
      : '未能获取真实列表：' + (d.error || '未知原因'), d.source !== 'provider');
  } catch (e) { toast('拉取失败：' + e.message, true); }
}

$('#set-fetch-chat').addEventListener('click', () => fetchModels('chat'));
$('#set-fetch-emb').addEventListener('click', () => fetchModels('embedding'));

$('#settings-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const payload = {
    profile: $('#set-profile').value || null,
    chat_model: $('#set-chat-model').value.trim() || null,
    base_url: $('#set-base-url').value.trim() || null,
    api_key: $('#set-api-key').value.trim() || null,
    embedding_model: $('#set-embedding-model').value.trim() || null,
    embedding_base_url: $('#set-embedding-base').value.trim() || null,
    embedding_api_key: $('#set-embedding-key').value.trim() || null,
  };
  if (!payload.embedding_base_url && !payload.embedding_api_key) {
    payload.clear = ['embedding_base_url', 'embedding_api_key'];
  }
  try {
    const d = await api('/settings', { method: 'POST', body: payload });
    $('#settings-output').textContent = (d.msg || '已保存') + '\n\n' + summarizeSettings(d.settings || {});
    $('#set-api-key').value = '';
    $('#set-embedding-key').value = '';
    toast('设置已保存并立即生效');
  } catch (e) { toast(e.message, true); }
});

$('#settings-test').addEventListener('click', async () => {
  const out = $('#settings-output');
  out.textContent = '测试中…';
  const lines = [];
  for (const [kind, title] of [['chat', '对话模型'], ['embedding', '向量化模型']]) {
    try {
      const d = await api('/settings/test', { method: 'POST', body: { kind } });
      lines.push((d.ok ? '✅ ' : '❌ ') + title + '：' + (d.detail || ''));
      if (d.hint) lines.push('　　↳ ' + d.hint);
    } catch (e) {
      lines.push('❌ ' + title + '：' + e.message);
    }
  }
  out.textContent = lines.join('\n');
});

/* ------------------------------------------------------------------ *
 * 启动
 * ------------------------------------------------------------------ */
(async function boot() {
  await checkBackend();
  await loadKbList();
  setInterval(checkBackend, 30000);
})();
