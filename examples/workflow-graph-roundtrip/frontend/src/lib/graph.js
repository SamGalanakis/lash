import { layoutDocument } from './layout.js';

// Build SvelteFlow nodes + edges from a draft WorkflowDocument.
//
// The draft document is the source of truth (the host owns the mutable draft).
// Each SvelteFlow node's `data.node` is a live reference into the draft, so an
// edit inside a node component mutates the draft that Save serializes back.
// Positions come from auto-layout, overridden by any user-dragged position
// persisted out-of-band in localStorage.
export function buildFlow(doc, storedPositions, onDelete, onAddNode, onReorder) {
  const { positions, sizes, groupLayouts } = layoutDocument(doc);
  const nodeMap = new Map(doc.nodes.map((n) => [n.id, n]));

  // Emit parents before children — SvelteFlow requires it for nesting.
  const ordered = [];
  const seen = new Set();
  function emit(id) {
    if (seen.has(id)) return;
    const node = nodeMap.get(id);
    if (!node) return;
    seen.add(id);
    ordered.push(node);
    for (const grp of node.data.children ?? []) {
      for (const childId of grp.nodeIds) emit(childId);
    }
  }
  for (const id of doc.roots?.processes ?? []) emit(id);
  for (const id of doc.roots?.main ?? []) emit(id);
  for (const node of doc.nodes) emit(node.id); // defensive: any strays

  const flowNodes = ordered.map((node) => {
    const size = sizes.get(node.id) ?? { w: 260, h: 90 };
    const stored = storedPositions[node.id];
    const auto = positions.get(node.id) ?? { x: 0, y: 0 };
    const isContainer = (node.data.children ?? []).length > 0;
    return {
      id: node.id,
      type: isContainer ? 'container' : node.data.kind === 'opaque' ? 'opaque' : 'workflow',
      position: stored ? { x: stored.x, y: stored.y } : { x: auto.x, y: auto.y },
      parentId: node.parentId,
      extent: node.parentId ? 'parent' : undefined,
      width: size.w,
      height: size.h,
      draggable: true,
      selectable: true,
      data: {
        node,
        width: size.w,
        height: size.h,
        groups: groupLayouts.get(node.id) ?? null,
        onDelete,
        onAddNode,
        onReorder,
      },
      // Containers must sit behind their children.
      zIndex: isContainer ? 0 : 1,
    };
  });

  // Classify data edges by scope. A data edge is "same-scope" when its source
  // and target share the same parent container/process group (identical
  // parentId — two top-level nodes both read as `null` and count as same
  // scope). Same-scope data edges route cleanly under dagre and keep their
  // variable-name pill.
  //
  // Cross-scope data edges — a variable defined in one scope but consumed
  // inside a nested container (e.g. `state`, defined in the process body, read
  // inside the `while` body) — cannot be routed by the per-scope layout, so
  // SvelteFlow draws them as floating dashed lines whose label chips land on
  // top of node headers and borders. We do not render them at all: the
  // consuming node already displays the variable name in its own body, so no
  // information is lost and the graph reads cleanly.
  const parentOf = new Map(doc.nodes.map((n) => [n.id, n.parentId ?? null]));
  const flowEdges = [];
  for (const edge of doc.edges) {
    const isData = edge.data?.kind === 'data';
    if (isData && parentOf.get(edge.source) !== parentOf.get(edge.target)) {
      continue; // cross-scope data edge — drop it (see note above)
    }
    flowEdges.push({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: false,
      label: isData ? edge.data?.variable ?? 'data' : undefined,
      data: { kind: edge.data?.kind ?? 'sequence' },
      class: isData ? 'edge-data' : 'edge-sequence',
      zIndex: 2,
      markerEnd: { type: 'arrowclosed', width: 16, height: 16 },
    });
  }

  return { flowNodes, flowEdges };
}

// ---------------------------------------------------------------------------
// Adding nodes.
//
// New nodes are minted client-side with temporary `new:<n>` ids. The backend
// treats any posted flow node whose id is not in its stored baseline as NEW: it
// builds a fresh node from `data.kind` (+ `data.subkind` for containers) and the
// editable field text below, then re-projects and returns a canonical document
// with real ids — so these temp ids only ever live inside the draft.
// ---------------------------------------------------------------------------

let nodeCounter = 0;
let edgeCounter = 0;

function mintNodeId(taken) {
  let id;
  do {
    id = `new:${(nodeCounter += 1)}`;
  } while (taken.has(id));
  taken.add(id);
  return id;
}

function mintEdgeId(taken) {
  let id;
  do {
    id = `new-edge:${(edgeCounter += 1)}`;
  } while (taken.has(id));
  taken.add(id);
  return id;
}

// Editable field defaults per kind. Each must parse as Lashlang (the user edits
// after inserting). Containers additionally declare a `subkind` and seed exactly
// one default child so their body is never empty. Returns the `data` payload
// (minus `children`, which addNodeToDoc fills in once the child ids exist).
function defaultNodeData(kind) {
  const base = { kind, nameSource: 'derived' };
  switch (kind) {
    case 'call':
      return { ...base, title: 'call', expression: 'display.show_message({ text: "new message" })' };
    case 'effect':
      return { ...base, title: 'sleep', effect: 'sleep', expression: 'sleep for "250ms"' };
    case 'data':
      return { ...base, title: 'data', binding: 'value', expression: '0' };
    case 'state_update':
      return { ...base, title: 'update state.count', target: 'state.count', expression: 'state.count + 1' };
    case 'computation':
      // binding intentionally omitted (optional `let` name).
      return { ...base, title: 'computation', expression: '1 + 1' };
    case 'terminal':
      return { ...base, title: 'finish', expression: '0' };
    case 'if':
      return { ...base, kind: 'container', subkind: 'if', title: 'if', condition: 'true' };
    case 'while':
      // condition deliberately false so a pre-edit Run cannot infinite-loop.
      return { ...base, kind: 'container', subkind: 'while', title: 'while', condition: 'false' };
    case 'for':
      return { ...base, kind: 'container', subkind: 'for', title: 'for', binding: 'item', iterable: '[1, 2, 3]' };
    case 'comprehension':
      return {
        ...base,
        kind: 'container',
        subkind: 'comprehension',
        title: 'comprehension',
        binding: 'items',
        clauses: [{ kind: 'for', binding: 'x', iterable: '[1, 2, 3]' }],
      };
    default:
      return { ...base, title: kind };
  }
}

// The single child seeded into a fresh container's slot: a default `call`, except
// a comprehension's `element` slot which is a single value expression (`x`).
function seedChildFor(kind) {
  if (kind === 'comprehension') {
    // A bare expression projects to a `data` node (binding-less value).
    return { kind: 'data', title: 'x', nameSource: 'derived', expression: 'x' };
  }
  return defaultNodeData('call');
}

// Container slot layout per subkind: which slots exist and which one is seeded
// with the initial child. `else` is left empty; the comprehension `element` slot
// must hold exactly one node, so it is never an insertion target (see below).
function containerSlots(kind) {
  switch (kind) {
    case 'if':
      return [{ slot: 'then', seeded: true }];
    case 'while':
    case 'for':
      return [{ slot: 'body', seeded: true }];
    case 'comprehension':
      return [{ slot: 'element', seeded: true }];
    default:
      return [];
  }
}

// Resolve an insertion target to the concrete list its new node id joins, the
// parentId to stamp, and the scope string sequence edges in that group carry.
// A target is either { main: true } or { ownerId, slot }.
function resolveGroup(doc, target) {
  if (target?.main) {
    doc.roots ??= { main: [], processes: [] };
    doc.roots.main ??= [];
    return { nodeIds: doc.roots.main, parentId: undefined, scope: 'main' };
  }
  const owner = doc.nodes.find((n) => n.id === target.ownerId);
  const group = (owner?.data.children ?? []).find((g) => g.slot === target.slot);
  if (!group) return null;
  return { nodeIds: group.nodeIds, parentId: owner.id, scope: group.scope };
}

// Insert a new default node of `kind` into the target group. Mirrors
// deleteNodeFromDoc's draft-mutation style: splice the id into the group's
// nodeIds (BEFORE a trailing terminal so the new node isn't dead code after a
// `finish`/`fail`), push the FlowNode, set parentId to match how buildFlow reads
// membership, and chain sequence edges around the new position. Container kinds
// also get their slot group(s) + one seeded child. Data edges are never added
// here — the backend recomputes them on reproject. Mutates `doc`; returns the
// new node id (or null if the target could not be resolved).
export function addNodeToDoc(doc, target, kind) {
  const taken = new Set(doc.nodes.map((n) => n.id));
  const takenEdges = new Set(doc.edges.map((e) => e.id));
  const group = resolveGroup(doc, target);
  if (!group) return null;

  const id = mintNodeId(taken);
  const data = defaultNodeData(kind);

  // Seed container slot groups (+ their initial child nodes) before appending.
  if (data.kind === 'container') {
    data.children = [];
    for (const { slot, seeded } of containerSlots(kind)) {
      const nodeIds = [];
      if (seeded) {
        const childId = mintNodeId(taken);
        const childData = seedChildFor(kind);
        doc.nodes.push({
          id: childId,
          type: childData.kind,
          parentId: id,
          data: childData,
        });
        nodeIds.push(childId);
      }
      data.children.push({ slot, scope: `container:${id}:${slot}`, nodeIds });
    }
  }

  const flowNode = { id, type: data.kind, data };
  if (group.parentId !== undefined) flowNode.parentId = group.parentId;
  doc.nodes.push(flowNode);

  // Source order == nodeIds order (the backend renders the block in node order),
  // so a new node appended after a trailing `terminal` (`finish`/`fail`) would be
  // dead code. Insert BEFORE a trailing terminal; otherwise append.
  const ids = group.nodeIds;
  const kindOf = new Map(doc.nodes.map((n) => [n.id, n.data?.kind]));
  const lastId = ids[ids.length - 1];
  const trailingTerminal = lastId !== undefined && kindOf.get(lastId) === 'terminal';
  const insertIndex = trailingTerminal ? ids.length - 1 : ids.length;
  const before = insertIndex > 0 ? ids[insertIndex - 1] : undefined; // now precedes new
  const after = insertIndex < ids.length ? ids[insertIndex] : undefined; // now follows new
  ids.splice(insertIndex, 0, id);

  // Chain sequence edges so ordering is explicit (the backend recomputes them on
  // reproject, but keep the draft sane — no edge skipping the new node).
  if (before !== undefined && after !== undefined) {
    // Was before → after; drop that so we can splice new between them.
    doc.edges = doc.edges.filter(
      (e) =>
        !(
          e.data?.kind === 'sequence' &&
          e.data?.scope === group.scope &&
          e.source === before &&
          e.target === after
        ),
    );
  }
  if (before !== undefined) {
    doc.edges.push({
      id: mintEdgeId(takenEdges),
      source: before,
      target: id,
      data: { kind: 'sequence', scope: group.scope },
    });
  }
  if (after !== undefined) {
    doc.edges.push({
      id: mintEdgeId(takenEdges),
      source: id,
      target: after,
      data: { kind: 'sequence', scope: group.scope },
    });
  }
  return id;
}

// Remove a node from the draft document: drop it from nodes, from every root /
// child group that references it, and drop incident edges. Mutates `doc`.
export function deleteNodeFromDoc(doc, nodeId) {
  const node = doc.nodes.find((n) => n.id === nodeId);
  if (!node) return;
  // Collect the node and all its descendants (deleting a container prunes its subtree).
  const toRemove = new Set();
  const nodeMap = new Map(doc.nodes.map((n) => [n.id, n]));
  (function collect(id) {
    if (toRemove.has(id)) return;
    toRemove.add(id);
    const n = nodeMap.get(id);
    for (const grp of n?.data.children ?? []) for (const c of grp.nodeIds) collect(c);
  })(nodeId);

  doc.nodes = doc.nodes.filter((n) => !toRemove.has(n.id));
  doc.edges = doc.edges.filter((e) => !toRemove.has(e.source) && !toRemove.has(e.target));
  if (doc.roots) {
    doc.roots.main = (doc.roots.main ?? []).filter((id) => !toRemove.has(id));
    doc.roots.processes = (doc.roots.processes ?? []).filter((id) => !toRemove.has(id));
  }
  for (const n of doc.nodes) {
    for (const grp of n.data.children ?? []) {
      grp.nodeIds = grp.nodeIds.filter((id) => !toRemove.has(id));
    }
  }
}

// ---------------------------------------------------------------------------
// Reordering statements within a scope.
//
// Execution order == the order of ids in a group's `nodeIds`; the backend
// renders source from that order and reprojects. Locate the group that owns
// `nodeId` (the top-level `main` root or a container/process child group),
// then move the id within that list. Mirrors deleteNodeFromDoc's
// draft-mutation style. Sequence edges within the scope are re-chained locally
// for immediate visual correctness — the backend recomputes them regardless on
// reproject. Mutates `doc`; returns true if the order actually changed.
// ---------------------------------------------------------------------------

// Resolve the group list (and its sequence-edge scope) that contains `nodeId`.
// Top-level `main` nodes carry the `main` scope; processes are parallel
// top-level definitions with no sequence ordering (scope null, no re-chain).
function findGroupOf(doc, nodeId) {
  const main = doc.roots?.main ?? [];
  if (main.includes(nodeId)) return { nodeIds: main, scope: 'main' };
  const procs = doc.roots?.processes ?? [];
  if (procs.includes(nodeId)) return { nodeIds: procs, scope: null };
  for (const n of doc.nodes) {
    for (const grp of n.data.children ?? []) {
      if (grp.nodeIds.includes(nodeId)) return { nodeIds: grp.nodeIds, scope: grp.scope ?? null };
    }
  }
  return null;
}

// Drop every sequence edge in `scope` and re-chain consecutive members of the
// (already reordered) group so the drawn arrows match the new order.
function rechainSequenceEdges(doc, group) {
  const scope = group.scope;
  if (!scope) return;
  const ids = group.nodeIds;
  doc.edges = doc.edges.filter((e) => !(e.data?.kind === 'sequence' && e.data?.scope === scope));
  const takenEdges = new Set(doc.edges.map((e) => e.id));
  for (let i = 0; i + 1 < ids.length; i += 1) {
    doc.edges.push({
      id: mintEdgeId(takenEdges),
      source: ids[i],
      target: ids[i + 1],
      data: { kind: 'sequence', scope },
    });
  }
}

// Move `nodeId` within its scope. `direction` is 'up' | 'down' or a target
// index. A trailing terminal (`finish`/`fail`) is a barrier: a non-terminal
// node cannot move after it (clamp), and a terminal itself never moves.
export function reorderNodeInDoc(doc, nodeId, direction) {
  const group = findGroupOf(doc, nodeId);
  if (!group) return false;
  const ids = group.nodeIds;
  const from = ids.indexOf(nodeId);
  if (from === -1) return false;

  const kindOf = new Map(doc.nodes.map((n) => [n.id, n.data?.kind]));
  if (kindOf.get(nodeId) === 'terminal') return false; // terminals stay put

  // A trailing terminal reserves the last slot; movable nodes stop before it.
  const lastId = ids[ids.length - 1];
  const hasTrailingTerminal = lastId !== undefined && kindOf.get(lastId) === 'terminal';
  const maxIndex = hasTrailingTerminal ? ids.length - 2 : ids.length - 1;

  let to = direction === 'up' ? from - 1 : direction === 'down' ? from + 1 : Number(direction);
  if (Number.isNaN(to)) return false;
  to = Math.max(0, Math.min(to, maxIndex));
  if (to === from) return false;

  ids.splice(from, 1);
  ids.splice(to, 0, nodeId);
  rechainSequenceEdges(doc, group);
  return true;
}
