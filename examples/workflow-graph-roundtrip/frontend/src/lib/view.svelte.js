// Which surface the workflow is shown on. Orthogonal to the simplified/power
// MODE toggle (lib/mode.svelte.js): a view is the whole rendering surface, a
// mode is the affordance depth within it.
//
// STEPS (default) is the plain-language linear checklist for a non-technical
// person (FIG-391). CANVAS is the existing node/graph editor, kept for
// comparison. Both edit the SAME draft document + lens; switching never mutates
// anything.

const KEY = 'lash.wfgraph.view.v1';

function load() {
  try {
    return localStorage.getItem(KEY) === 'canvas' ? 'canvas' : 'steps';
  } catch {
    return 'steps';
  }
}

export class ViewController {
  current = $state(load());

  get steps() {
    return this.current === 'steps';
  }
  get canvas() {
    return this.current === 'canvas';
  }

  set(view) {
    this.current = view === 'canvas' ? 'canvas' : 'steps';
    try {
      localStorage.setItem(KEY, this.current);
    } catch {
      /* storage disabled — toggle just won't persist */
    }
  }
}
