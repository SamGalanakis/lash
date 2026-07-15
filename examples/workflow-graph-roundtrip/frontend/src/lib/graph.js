import { layoutDocument } from './layout.js';

// Build SvelteFlow nodes + edges from a draft WorkflowDocument.
//
// The draft document is the source of truth (the host owns the mutable draft).
// Each SvelteFlow node's `data.node` is a live reference into the draft, so an
// edit inside a node component mutates the draft that Save serializes back.
// Positions come from auto-layout, overridden by any user-dragged position
// persisted out-of-band in localStorage.
export function buildFlow(doc, storedPositions, onDelete) {
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
