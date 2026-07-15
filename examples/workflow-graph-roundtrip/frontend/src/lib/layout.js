import dagre from '@dagrejs/dagre';

// Auto-layout for a positionless WorkflowDocument.
//
// The document is a compound graph: a process contains its body, an `if`
// container contains `then`/`else` groups, etc. We lay it out post-order so a
// container's size is known from its already-laid-out children, then run dagre
// per scope to flow nodes top-to-bottom along their edges. All coordinates are
// returned RELATIVE TO EACH NODE'S PARENT, which is exactly what SvelteFlow's
// nested-node coordinate system expects.

const NODE_W = 264;
const OPAQUE_W = 340;
const SMALL_W = 208;
const HEADER_BAND = 46; // container title band
const GROUP_LABEL = 24; // slot label (then/else/body) height
const PAD = 20;
const GROUP_GAP = 30;
const NODESEP = 34;
const RANKSEP = 52;

export function layoutDocument(doc) {
  const nodeMap = new Map(doc.nodes.map((n) => [n.id, n]));
  const sizes = new Map();
  const positions = new Map();
  const groupLayouts = new Map();

  function estimateLeaf(node) {
    const kind = node.data.kind;
    let w = NODE_W;
    if (kind === 'opaque') w = OPAQUE_W;
    else if (kind === 'terminal' || kind === 'data') w = SMALL_W;

    let h = 56; // kind badge + title
    if (node.data.description) h += 20;
    if (kind === 'opaque') {
      const lines = (node.data.source ?? '').split('\n').length;
      h += Math.min(Math.max(lines, 3), 16) * 17 + 34;
    } else {
      const fieldCount = Object.keys(node.data.fields ?? {}).length;
      h += fieldCount * 48;
      // Typed assignment / computation nodes render dedicated expression rows
      // that are not part of `fields`, so reserve space for them here.
      if (kind === 'state_update') h += 52; // target ≔ expression
      else if (kind === 'computation') h += 84; // optional binding + expression
    }
    h += 26; // footer / delete affordance
    return { w, h: Math.max(h, 82) };
  }

  function isContainer(node) {
    return (node.data.children ?? []).length > 0;
  }

  // Lay out one scope (an ordered list of node ids) with dagre, returning
  // positions relative to the scope's own top-left origin, plus its bbox.
  function layoutScope(nodeIds) {
    if (!nodeIds || nodeIds.length === 0) {
      return { positions: new Map(), width: 120, height: 40 };
    }
    // Post-order: size children first.
    for (const id of nodeIds) {
      const node = nodeMap.get(id);
      if (!node) continue;
      if (isContainer(node)) sizeContainer(node);
      else if (!sizes.has(id)) sizes.set(id, estimateLeaf(node));
    }

    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: 'TB', nodesep: NODESEP, ranksep: RANKSEP, marginx: 6, marginy: 6 });
    g.setDefaultEdgeLabel(() => ({}));
    const idSet = new Set(nodeIds);
    for (const id of nodeIds) {
      const s = sizes.get(id) ?? { w: NODE_W, h: 90 };
      g.setNode(id, { width: s.w, height: s.h });
    }
    for (const edge of doc.edges) {
      if (idSet.has(edge.source) && idSet.has(edge.target)) {
        g.setEdge(edge.source, edge.target, { weight: edge.data?.kind === 'data' ? 1 : 3 });
      }
    }
    dagre.layout(g);

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const local = new Map();
    for (const id of nodeIds) {
      const gn = g.node(id);
      if (!gn) continue;
      const x = gn.x - gn.width / 2;
      const y = gn.y - gn.height / 2;
      local.set(id, { x, y, w: gn.width, h: gn.height });
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + gn.width);
      maxY = Math.max(maxY, y + gn.height);
    }
    const rel = new Map();
    for (const [id, p] of local) rel.set(id, { x: p.x - minX, y: p.y - minY });
    return { positions: rel, width: maxX - minX, height: maxY - minY };
  }

  // Size a container/process node from its child groups and set the absolute
  // (parent-relative) positions of every descendant it directly frames.
  function sizeContainer(node) {
    if (sizes.has(node.id) && groupLayouts.has(node.id)) return;
    const groups = node.data.children ?? [];
    const laid = groups.map((grp) => ({ slot: grp.slot, ...layoutScope(grp.nodeIds) }));

    let x = PAD;
    let maxH = 0;
    for (const lg of laid) {
      lg.offsetX = x;
      lg.offsetY = HEADER_BAND + GROUP_LABEL;
      x += lg.width + GROUP_GAP;
      maxH = Math.max(maxH, lg.height);
    }
    const innerRight = laid.length ? x - GROUP_GAP + PAD : PAD * 3;
    const totalW = Math.max(innerRight, 220);
    const totalH = HEADER_BAND + (laid.length ? GROUP_LABEL + maxH + PAD : PAD);

    for (const lg of laid) {
      for (const [id, p] of lg.positions) {
        positions.set(id, { x: lg.offsetX + p.x, y: lg.offsetY + p.y });
      }
    }
    sizes.set(node.id, { w: totalW, h: totalH });
    groupLayouts.set(
      node.id,
      laid.map((lg) => ({
        slot: lg.slot,
        x: lg.offsetX,
        y: HEADER_BAND,
        w: Math.max(lg.width, 80),
        h: GROUP_LABEL + lg.height,
      })),
    );
  }

  // Top level: processes (and any bare main nodes) placed left-to-right.
  const topIds = [...(doc.roots?.processes ?? [])];
  for (const id of doc.roots?.main ?? []) {
    const n = nodeMap.get(id);
    if (n && !n.parentId) topIds.push(id);
  }
  for (const id of topIds) {
    const node = nodeMap.get(id);
    if (!node) continue;
    if (isContainer(node)) sizeContainer(node);
    else if (!sizes.has(id)) sizes.set(id, estimateLeaf(node));
  }
  let tx = 0;
  for (const id of topIds) {
    positions.set(id, { x: tx, y: 0 });
    tx += (sizes.get(id)?.w ?? NODE_W) + 80;
  }

  // Any orphan nodes not reached above (defensive) get a fallback slot.
  let oy = 0;
  for (const node of doc.nodes) {
    if (!positions.has(node.id)) {
      if (!sizes.has(node.id)) sizes.set(node.id, estimateLeaf(node));
      positions.set(node.id, { x: tx + 40, y: oy });
      oy += (sizes.get(node.id)?.h ?? 90) + 30;
    }
  }

  return { positions, sizes, groupLayouts };
}
