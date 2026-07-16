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

// Rewrite persisted position keys through a Save-response `idMap`
// ({ "<oldId>": "<newId>", ... }). Node ids are remade on every Save, so a
// dragged node's stored position would otherwise be orphaned under its old id.
// Only keys present in the map are remapped; positions belonging to other
// workflows (and to nodes that were deleted, hence absent from the map) are
// left untouched — the orphaned old key simply never gets referenced again.
export function migratePositions(idMap) {
  if (!idMap) return;
  const all = readAll();
  for (const [oldId, newId] of Object.entries(idMap)) {
    if (oldId === newId) continue;
    if (Object.prototype.hasOwnProperty.call(all, oldId)) {
      all[newId] = all[oldId];
      delete all[oldId];
    }
  }
  writeAll(all);
}
