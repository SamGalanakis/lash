(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const entries = $$('.entry');
  const ticks = $$('.spine-tick');
  const tickById = new Map(ticks.map((t) => [t.getAttribute('href').slice(1), t]));
  const usageRows = $$('.usage-row');
  const usageById = new Map(usageRows.map((r) => [r.getAttribute('href').slice(1), r]));
  const usageOverviewBars = $$('.usage-overview-bar');
  const overviewById = new Map();
  const transcript = $('#transcript');
  const usageChart = $('.usage-chart');

  usageOverviewBars.forEach((bar, idx) => {
    const row = usageRows[idx];
    if (!row) return;
    const href = row.getAttribute('href');
    const id = href.slice(1);
    bar.setAttribute('href', href);
    bar.dataset.usageEntry = id;
    overviewById.set(id, bar);
  });

  let layoutQueued = false;
  function queueUsageLayout() {
    if (layoutQueued) return;
    layoutQueued = true;
    requestAnimationFrame(() => {
      layoutQueued = false;
      layoutUsageRows();
    });
  }

  function layoutUsageRows() {
    if (!usageChart || !transcript || !usageRows.length) return;
    const chartTop = usageChart.getBoundingClientRect().top + window.scrollY;
    usageChart.style.height = `${transcript.offsetHeight}px`;
    for (const row of usageRows) {
      const id = row.getAttribute('href').slice(1);
      const entry = document.getElementById(id);
      if (!entry) continue;
      const entryTop = entry.getBoundingClientRect().top + window.scrollY;
      row.style.top = `${Math.max(0, entryTop - chartTop)}px`;
    }
  }

  // ─── filters ────────────────────────────────────────────────────────────

  const state = {
    role: new Set(['user', 'assistant', 'tool', 'rlm', 'llm_call', 'system']),
    tool: new Set(),
    toolKnown: new Set(),
    search: '',
  };

  // initialize the tool set from the chip row
  $$('.chip[data-filter="tool"]').forEach((chip) => {
    const v = chip.dataset.value;
    state.tool.add(v);
    if (v !== '__other__') state.toolKnown.add(v);
  });

  function applyFilters() {
    const q = state.search.trim().toLowerCase();
    let visible = 0;
    let hits = 0;
    for (const entry of entries) {
      const role = entry.dataset.role;
      const tool = entry.dataset.tool;
      let hidden = false;
      if (!state.role.has(role)) hidden = true;
      if (!hidden && role === 'tool') {
        if (state.toolKnown.has(tool)) {
          if (!state.tool.has(tool)) hidden = true;
        } else {
          if (!state.tool.has('__other__')) hidden = true;
        }
      }
      if (!hidden && q) {
        const hay = entry.dataset.search || '';
        if (!hay.includes(q)) hidden = true;
        else hits++;
      }
      entry.classList.toggle('is-hidden', hidden);
      if (!hidden) visible++;
      const tick = tickById.get(entry.id);
      if (tick) tick.classList.toggle('is-hidden', hidden);
      const usage = usageById.get(entry.id);
      if (usage) usage.classList.toggle('is-hidden', hidden);
      const overview = overviewById.get(entry.id);
      if (overview) overview.classList.toggle('is-hidden', hidden);
    }
    const meta = $('#q-meta');
    if (meta) {
      if (q) meta.textContent = `${hits} match${hits === 1 ? '' : 'es'}`;
      else meta.textContent = `${visible} / ${entries.length}`;
    }
    let empty = $('#filter-empty');
    if (!empty && transcript) {
      empty = document.createElement('div');
      empty.id = 'filter-empty';
      empty.className = 'filter-empty';
      empty.innerHTML = `<div class="filter-empty-text"></div><button type="button" class="chip" id="filter-empty-reset">reset filters</button>`;
      transcript.appendChild(empty);
      const reset = empty.querySelector('#filter-empty-reset');
      if (reset) {
        reset.addEventListener('click', () => {
          state.role = new Set(['user', 'assistant', 'tool', 'rlm', 'llm_call', 'system']);
          state.tool = new Set(state.toolKnown);
          if ($$('.chip[data-filter="tool"][data-value="__other__"]').length) state.tool.add('__other__');
          state.search = '';
          const input = $('#q'); if (input) input.value = '';
          $$('.chip[data-filter]').forEach((c) => c.classList.add('is-on'));
          $$('.chip[data-toggle]').forEach((c) => c.classList.remove('is-on'));
          applyFilters();
        });
      }
    }
    if (empty) {
      const showEmpty = entries.length > 0 && visible === 0;
      empty.classList.toggle('is-on', showEmpty);
      if (showEmpty) {
        const txt = empty.querySelector('.filter-empty-text');
        if (txt) txt.textContent = `0 of ${entries.length} entries match the current filters.`;
      }
    }
    if (q) highlightMatches(q);
    else clearHighlights();
    queueUsageLayout();
  }

  // ─── search highlight ──────────────────────────────────────────────────

  function clearHighlights() {
    $$('.entry mark').forEach((m) => {
      const parent = m.parentNode;
      parent.replaceChild(document.createTextNode(m.textContent), m);
      parent.normalize();
    });
  }

  function highlightMatches(q) {
    clearHighlights();
    if (!q) return;
    const lc = q.toLowerCase();
    for (const entry of entries) {
      if (entry.classList.contains('is-hidden')) continue;
      const targets = entry.querySelectorAll(
        '.entry-headline, .part--prose, .reasoning-text, .code-pre, .json'
      );
      targets.forEach((node) => highlightInNode(node, lc));
    }
  }

  function highlightInNode(node, lc) {
    const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null);
    const matches = [];
    let textNode;
    while ((textNode = walker.nextNode())) {
      const text = textNode.nodeValue;
      const lower = text.toLowerCase();
      let i = 0;
      let pos;
      while ((pos = lower.indexOf(lc, i)) !== -1) {
        matches.push({ node: textNode, start: pos, end: pos + lc.length });
        i = pos + lc.length;
      }
    }
    // process in reverse so earlier offsets stay valid
    for (let m = matches.length - 1; m >= 0; m--) {
      const { node, start, end } = matches[m];
      const after = node.splitText(end);
      const matchNode = node.splitText(start);
      const mark = document.createElement('mark');
      mark.textContent = matchNode.nodeValue;
      matchNode.parentNode.replaceChild(mark, matchNode);
      void after; // referenced for clarity
    }
  }

  // ─── chip wiring ────────────────────────────────────────────────────────

  $$('.chip[data-filter="role"]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const v = chip.dataset.value;
      if (state.role.has(v)) state.role.delete(v);
      else state.role.add(v);
      chip.classList.toggle('is-on');
      applyFilters();
    });
  });

  $$('.chip[data-filter="tool"]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const v = chip.dataset.value;
      if (state.tool.has(v)) state.tool.delete(v);
      else state.tool.add(v);
      chip.classList.toggle('is-on');
      applyFilters();
    });
  });

  $$('.chip[data-toggle]').forEach((chip) => {
    chip.addEventListener('click', () => {
      const t = chip.dataset.toggle;
      const on = chip.classList.toggle('is-on');
      document.body.classList.toggle(t, on);
    });
  });

  $$('.chip[data-action="expand-all"]').forEach((chip) => {
    chip.addEventListener('click', () => expandAll(true));
  });
  $$('.chip[data-action="collapse-all"]').forEach((chip) => {
    chip.addEventListener('click', () => expandAll(false));
  });

  function expandAll(open) {
    $$('details').forEach((d) => {
      d.open = open;
    });
    queueUsageLayout();
  }

  // ─── search input ───────────────────────────────────────────────────────

  const q = $('#q');
  let searchTimer = null;
  q.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.search = q.value;
      applyFilters();
    }, 70);
  });

  // ─── copy buttons ───────────────────────────────────────────────────────

  document.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-copy]');
    if (!btn) return;
    ev.preventDefault();
    const block = btn.closest('.kv, .part--code, .part--output, .part--error, .part--final, details');
    const target = block ? block.querySelector('pre') : null;
    if (!target) return;
    const text = target.textContent;
    navigator.clipboard.writeText(text).then(() => {
      const prev = btn.textContent;
      btn.textContent = 'copied';
      btn.classList.add('is-copied');
      setTimeout(() => {
        btn.textContent = prev;
        btn.classList.remove('is-copied');
      }, 900);
    });
  });

  // ─── permalink anchors flash on click ───────────────────────────────────

  document.addEventListener('click', (ev) => {
    const a = ev.target.closest('.entry-num');
    if (!a) return;
    // entry-num may live inside a <summary>; without
    // stopPropagation the click would also toggle the parent <details>.
    ev.stopPropagation();
    const id = a.getAttribute('href').slice(1);
    flashEntry(id);
  });


  function flashEntry(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('is-flash');
    void el.offsetWidth;
    el.classList.add('is-flash');
  }

  // ─── spine sync (current entry highlight) ──────────────────────────────

  let currentTick = null;
  let currentUsage = null;
  let currentOverview = null;
  const setCurrent = (id) => {
    if (currentTick) currentTick.classList.remove('is-current');
    if (currentUsage) currentUsage.classList.remove('is-current');
    if (currentOverview) currentOverview.classList.remove('is-current');
    const t = tickById.get(id);
    if (t) {
      t.classList.add('is-current');
      currentTick = t;
    }
    const u = usageById.get(id);
    if (u) {
      u.classList.add('is-current');
      currentUsage = u;
    }
    const o = overviewById.get(id);
    if (o) {
      o.classList.add('is-current');
      currentOverview = o;
    }
  };

  if (entries.length && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (records) => {
        const visible = records
          .filter((r) => r.isIntersecting)
          .sort((a, b) => a.target.offsetTop - b.target.offsetTop);
        if (visible[0]) setCurrent(visible[0].target.id);
      },
      { rootMargin: '-72px 0px -60% 0px', threshold: [0, 0.25, 0.5] }
    );
    entries.forEach((e) => io.observe(e));
  }

  // ─── keyboard navigation ───────────────────────────────────────────────

  function visibleEntries() {
    return entries.filter((e) => !e.classList.contains('is-hidden'));
  }

  function currentIndex(list) {
    if (!currentTick) return -1;
    const id = currentTick.getAttribute('href').slice(1);
    return list.findIndex((e) => e.id === id);
  }

  function jumpTo(el) {
    if (!el) return;
    // open the entry's primary <details> so j/k navigation actually shows
    // content rather than parking on a closed header
    const det = el.querySelector(':scope > .entry-body > details');
    if (det && !det.open) det.open = true;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    queueUsageLayout();
    setCurrent(el.id);
    flashEntry(el.id);
  }

  // open details when arriving via hash (#e23)
  function expandHashTarget() {
    const id = location.hash.slice(1);
    if (!id) return;
    const el = document.getElementById(id);
    if (!el) return;
    const det = el.querySelector(':scope > .entry-body > details');
    if (det && !det.open) det.open = true;
    flashEntry(id);
  }
  window.addEventListener('hashchange', expandHashTarget);
  window.addEventListener('resize', queueUsageLayout);
  document.addEventListener('toggle', queueUsageLayout, true);
  queueUsageLayout();
  if (location.hash) setTimeout(expandHashTarget, 0);

  document.addEventListener('keydown', (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) {
      if (ev.key === 'Escape') {
        ev.target.blur();
        if (q.value) {
          q.value = '';
          state.search = '';
          applyFilters();
        }
      }
      return;
    }
    const list = visibleEntries();
    if (ev.key === 'j' || ev.key === 'ArrowDown') {
      ev.preventDefault();
      const i = currentIndex(list);
      jumpTo(list[Math.min(list.length - 1, i + 1)]);
    } else if (ev.key === 'k' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      const i = currentIndex(list);
      jumpTo(list[Math.max(0, i - 1)]);
    } else if (ev.key === 'Home') {
      ev.preventDefault();
      jumpTo(list[0]);
    } else if (ev.key === 'End') {
      ev.preventDefault();
      jumpTo(list[list.length - 1]);
    } else if (ev.key === 'e') {
      expandAll(true);
    } else if (ev.key === 'c') {
      expandAll(false);
    } else if (ev.key === '/') {
      ev.preventDefault();
      q.focus();
      q.select();
    } else if (ev.key === '?') {
      ev.preventDefault();
      toggleHelp(true);
    } else if (ev.key === 'Escape') {
      toggleHelp(false);
    }
  });

  // ─── help overlay ──────────────────────────────────────────────────────

  const help = document.createElement('div');
  help.className = 'help-overlay';
  help.innerHTML = `
    <div class="help-card">
      <h2>Keyboard</h2>
      <dl>
        <dt><kbd>j</kbd>/<kbd>k</kbd></dt><dd>next / previous entry</dd>
        <dt><kbd>Home</kbd>/<kbd>End</kbd></dt><dd>jump to first / last</dd>
        <dt><kbd>e</kbd>/<kbd>c</kbd></dt><dd>expand / collapse all</dd>
        <dt><kbd>/</kbd></dt><dd>focus search</dd>
        <dt><kbd>Esc</kbd></dt><dd>clear search · close help</dd>
        <dt><kbd>?</kbd></dt><dd>toggle this help</dd>
      </dl>
      <h2>Terms</h2>
      <dl class="help-terms">
        <dt>llm call</dt><dd>one outbound request to the model. Renders the system block and request messages.</dd>
        <dt>tool call</dt><dd>the agent invoked a registered tool. The result is rendered inline.</dd>
        <dt>direct completion</dt><dd>an LLM call issued from inside a tool (e.g. tournament_rerank's batch reranks). Folded under the parent tool call.</dd>
        <dt>lashlang step</dt><dd>one execution entry from recursive language-model mode: model emits a lashlang block, runtime executes it.</dd>
        <dt>system hash</dt><dd>short fingerprint of the system prompt; identical hashes mean the same system text. Repeats are coalesced.</dd>
      </dl>
      <div class="help-foot">click a chip to filter · click an entry's id (e.g. e23) to copy a permalink</div>
    </div>`;
  document.body.appendChild(help);
  help.addEventListener('click', (ev) => {
    if (ev.target === help) toggleHelp(false);
  });
  function toggleHelp(open) {
    help.classList.toggle('is-open', open);
  }

  // ─── initial pass ──────────────────────────────────────────────────────

  applyFilters();
})();

// ─── multi-view: lineage / breadcrumb / browser history ───────────────────

(function () {
  var tree = window.__lashTraceTree;
  if (!tree || !tree.length) return;

  var byId = Object.create(null);
  tree.forEach(function (n) { byId[n.id] = n; });

  var trail = document.getElementById('crumb-trail');
  var backBtn = document.getElementById('back-btn');
  var views = Array.prototype.slice.call(document.querySelectorAll('.view'));

  function ancestors(id) {
    var chain = [];
    var cur = byId[id];
    while (cur) { chain.unshift(cur); cur = cur.parent ? byId[cur.parent] : null; }
    return chain;
  }

  function renderCrumb(id) {
    var chain = ancestors(id);
    if (!chain.length) return;
    while (trail.children.length > 1) trail.removeChild(trail.lastChild);
    chain.forEach(function (step, i) {
      if (i > 0) {
        var sep = document.createElement('span');
        sep.className = 'crumb-sep';
        sep.textContent = '›';
        trail.appendChild(sep);
      }
      var btn = document.createElement('button');
      btn.className = 'crumb-step';
      btn.dataset.go = step.id;
      btn.innerHTML = escapeHtml(step.label) + ' <span class="crumb-id">' + escapeHtml(step.sid) + '</span>';
      if (i === chain.length - 1) btn.setAttribute('aria-current', 'true');
      trail.appendChild(btn);
    });
    backBtn.disabled = chain.length <= 1;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function goTo(id, push) {
    if (!byId[id]) return;
    views.forEach(function (v) { v.classList.toggle('is-active', v.dataset.view === id); });
    renderCrumb(id);
    if (push !== false && location.hash !== '#' + id) {
      history.pushState({ view: id }, '', '#' + id);
    }
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  document.addEventListener('click', function (ev) {
    var t = ev.target.closest && ev.target.closest('[data-go]');
    if (!t) return;
    if (!byId[t.dataset.go]) return;
    ev.preventDefault();
    goTo(t.dataset.go, true);
  });

  if (backBtn) {
    backBtn.addEventListener('click', function () {
      var cur = views.find(function (v) { return v.classList.contains('is-active'); });
      if (!cur) return;
      var node = byId[cur.dataset.view];
      if (node && node.parent) goTo(node.parent, true);
    });
  }

  window.addEventListener('popstate', function (ev) {
    var v = (ev.state && ev.state.view) || (location.hash.replace(/^#/, '') || 'root');
    goTo(v, false);
  });

  var initial = location.hash.replace(/^#/, '') || 'root';
  if (byId[initial]) {
    goTo(initial, false);
  } else {
    renderCrumb('root');
  }
})();
