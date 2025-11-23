// Renderer helpers for GridWorld UI
// Exports:
//  - renderGrid(world, gridEl)
//  - renderAgent(world, gridEl)
//  - animateMove(oldState, newState, gridEl)
//  - highlightGoal(world, gridEl)
//  - highlightObstacles(world, gridEl)

export function renderGrid(world, gridEl){
  const n = world.size;
  gridEl.innerHTML = '';
  const cellSize = getComputedStyle(document.documentElement).getPropertyValue('--cell-size').trim() || '44px';
  gridEl.style.gridTemplateColumns = `repeat(${n}, ${cellSize})`;
  for (let r=0;r<n;r++){
    for (let c=0;c<n;c++){
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.r = r; cell.dataset.c = c;
      if (world.isObstacle(r,c)) cell.classList.add('obstacle');
      else cell.classList.add('empty');
      if (r===world.start.r && c===world.start.c) cell.classList.add('start');
      if (r===world.goal.r && c===world.goal.c) cell.classList.add('goal');
      if (r===world.agent.r && c===world.agent.c) cell.classList.add('agent');
      gridEl.appendChild(cell);
    }
  }
}

// Render a policy overlay given a Q-table (2D array Q[state][action])
export function renderPolicy(world, gridEl, Q){
  // remove any existing policy spans
  gridEl.querySelectorAll('.policy').forEach(e=>e.remove());
  if (!Q) return;
  const n = world.size;
  const actions = ['↑','→','↓','←'];
  for (let s=0;s<Math.min(Q.length, n*n);s++){
    const rc = world.stateToRC(s);
    const q = Q[s] || [0,0,0,0];
    // pick argmax (ties arbitrary)
    let max = -Infinity; let best = 0;
    for (let a=0;a<q.length;a++){ if (q[a] > max){ max=q[a]; best=a; } }
    const cell = gridEl.querySelector(`[data-r='${rc.r}'][data-c='${rc.c}']`);
    if (cell){
      const span = document.createElement('div');
      span.className = 'policy';
      span.textContent = actions[best] || '';
      cell.appendChild(span);
    }
  }
}

// Render a value-function heatmap (V(s)=max_a Q[s][a]) as a colored overlay
export function renderValueHeatmap(world, gridEl, Q){
  gridEl.querySelectorAll('.heat').forEach(e=>e.remove());
  if (!Q) return;
  const n = world.size;
  const V = [];
  for (let s=0;s<Math.min(Q.length, n*n); s++){
    const q = Q[s] || [0,0,0,0];
    V.push(Math.max(...q));
  }
  if (V.length===0) return;
  const vmin = Math.min(...V); const vmax = Math.max(...V);
  const norm = v => (vmax===vmin) ? 0.5 : (v - vmin) / (vmax - vmin);
  for (let s=0;s<V.length;s++){
    const rc = world.stateToRC(s);
    const val = V[s];
    const t = norm(val);
    // color from cool (blue) to warm (red)
    const r = Math.round(255 * t);
    const g = Math.round(120 * (1 - Math.abs(t-0.5)*2));
    const b = Math.round(255 * (1 - t));
    const hex = `rgba(${r},${g},${b},0.6)`;
    const cell = gridEl.querySelector(`[data-r='${rc.r}'][data-c='${rc.c}']`);
    if (cell){
      const div = document.createElement('div');
      div.className = 'heat';
      div.style.background = hex;
      cell.appendChild(div);
    }
  }
}

export function renderAgent(world, gridEl){
  // remove any existing agent classes, then add to current agent cell
  const prev = gridEl.querySelectorAll('.cell.agent');
  prev.forEach(el=>el.classList.remove('agent'));
  const sel = `[data-r='${world.agent.r}'][data-c='${world.agent.c}']`;
  const cell = gridEl.querySelector(sel);
  if (cell) cell.classList.add('agent');
}

export function animateMove(oldState, newState, gridEl){
  // Basic CSS-based animation: remove agent from old, add to new with temporary class
  const oldSel = `[data-r='${oldState.r}'][data-c='${oldState.c}']`;
  const newSel = `[data-r='${newState.r}'][data-c='${newState.c}']`;
  const oldCell = gridEl.querySelector(oldSel);
  const newCell = gridEl.querySelector(newSel);
  if (oldCell) oldCell.classList.remove('agent');
  if (newCell){
    newCell.classList.add('agent');
    newCell.classList.add('move');
    // remove temporary animation class after transition
    setTimeout(()=>newCell.classList.remove('move'), 220);
  }
}

export function highlightGoal(world, gridEl){
  const sel = `[data-r='${world.goal.r}'][data-c='${world.goal.c}']`;
  const cell = gridEl.querySelector(sel);
  if (cell) cell.classList.add('goal');
}

export function highlightObstacles(world, gridEl){
  for (let r=0;r<world.size;r++){
    for (let c=0;c<world.size;c++){
      if (world.isObstacle(r,c)){
        const sel = `[data-r='${r}'][data-c='${c}']`;
        const cell = gridEl.querySelector(sel);
        if (cell) cell.classList.add('obstacle');
      }
    }
  }
}
