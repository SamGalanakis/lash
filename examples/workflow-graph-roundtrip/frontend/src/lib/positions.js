// Out-of-band canvas layout. Positions are the FRONTEND's business: the
// WorkflowDocument carries none, and we never send them back. We persist
// user-dragged positions in localStorage keyed by node id (positions are
// relative to a node's parent, matching SvelteFlow's coordinate system).

const KEY = 'lash.wfgraph.positions.v1';

function readAll() {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}');
  } catch {
    return {};
  }
}

function writeAll(map) {
  try {
    localStorage.setItem(KEY, JSON.stringify(map));
  } catch {
    /* storage full or unavailable — layout simply won't persist */
  }
}

export function loadPositions() {
  return readAll();
}

export function savePosition(nodeId, position) {
  const all = readAll();
  all[nodeId] = { x: Math.round(position.x), y: Math.round(position.y) };
  writeAll(all);
}

export function clearPositions() {
  writeAll({});
}
