import { GridWorld } from './env/gridworld.js';
import { renderGrid, renderAgent, animateMove, highlightGoal, highlightObstacles, renderPolicy, renderValueHeatmap } from './renderer.js';
import QLearningAgent from './agents/qlearning.js';
import SARSAAgent from './agents/sarsa.js';
import MonteCarloAgent from './agents/montecarlo.js';

let world = new GridWorld(5);
let running = false;
let paused = false;

const gridEl = document.getElementById('grid');
const sizeSelect = document.getElementById('sizeSelect');
const resetBtn = document.getElementById('resetBtn');
const stepBtn = document.getElementById('stepBtn');
const startBtn = document.getElementById('startBtn');
const pauseBtn = document.getElementById('pauseBtn');
const episodesInput = document.getElementById('episodesInput');
const algoSelect = document.getElementById('algoSelect');
const alphaSlider = document.getElementById('alpha');
const gammaSlider = document.getElementById('gamma');
const epsilonSlider = document.getElementById('epsilon');
const statusEl = document.getElementById('status');
const alphaVal = document.getElementById('alphaVal');
const gammaVal = document.getElementById('gammaVal');
const epsilonVal = document.getElementById('epsilonVal');
const headlessCheckbox = document.getElementById('headlessMode');
const showPolicyCheckbox = document.getElementById('showPolicy');
const showHeatmapCheckbox = document.getElementById('showHeatmap');
const editMazeCheckbox = document.getElementById('editMaze');
const exportBtn = document.getElementById('exportBtn');
let lastRunResults = null;

// charts
const rewardCtx = document.getElementById('rewardChart').getContext('2d');
const stepsCtx = document.getElementById('stepsChart').getContext('2d');
const convCtx = document.getElementById('convChart').getContext('2d');

const rewardChart = new Chart(rewardCtx, {
	type: 'line',
	data: {labels:[], datasets:[{label:'Episode reward',borderColor:'#2b8cff',data:[],fill:false}]},
	options:{responsive:true,maintainAspectRatio:false}
});

const stepsChart = new Chart(stepsCtx, {
	type: 'line',
	data: {labels:[], datasets:[{label:'Steps',borderColor:'#2ecc71',data:[],fill:false}]},
	options:{responsive:true,maintainAspectRatio:false}
});

const convChart = new Chart(convCtx, {
	type: 'line',
	data: {labels:[], datasets:[{label:'Avg reward (window 10)',borderColor:'#f39c12',data:[],fill:false}]},
	options:{responsive:true,maintainAspectRatio:false}
});

// comparison charts (initialized later)
const cmpRewardCtx = document.getElementById('cmpRewardChart')?.getContext('2d');
const cmpStepsCtx = document.getElementById('cmpStepsChart')?.getContext('2d');
const cmpConvergeCtx = document.getElementById('cmpConvergeChart')?.getContext('2d');
const cmpPathCtx = document.getElementById('cmpPathChart')?.getContext('2d');

let cmpRewardChart = null;
let cmpStepsChart = null;
let cmpConvergeChart = null;
let cmpPathChart = null;

// rendering delegated to renderer.js

function reset(size) {
	const s = size || Number(sizeSelect.value);
	world.reset(s);
	renderGrid(world, gridEl);
	// update overlays if any
	updateOverlays();
}

function randomAction(){
	return Math.floor(Math.random()*4);
}

async function runEpisodes(totalEpisodes){
	running = true; paused = false;
	startBtn.disabled = true; pauseBtn.disabled = false; resetBtn.disabled = true; stepBtn.disabled = true; sizeSelect.disabled = true;

	const rewardsHistory = [];
	const reachedHistory = [];
	const stepsHistory = [];
	let converged = false;

	// create agent based on selection and slider params
	const states = Number(sizeSelect.value) * Number(sizeSelect.value);
	const actions = 4;
	const params = { alpha: Number(alphaSlider.value), gamma: Number(gammaSlider.value), epsilon: Number(epsilonSlider.value) };
	const headless = !!(headlessCheckbox && headlessCheckbox.checked);
	const animateEnabled = !headless;
	let agent = null;
	const algo = algoSelect.value;
	if (algo === 'qlearning') agent = new QLearningAgent({states, actions, ...params});
	else if (algo === 'sarsa') agent = new SARSAAgent({states, actions, ...params});
	else agent = new MonteCarloAgent({states, actions, ...params});

	// expose to renderer overlays
	if (window._setCurrentAgent) window._setCurrentAgent(agent);

	// prepare storage for export / analysis
	lastRunResults = { algorithm: algo, episodes: totalEpisodes, alpha: params.alpha, gamma: params.gamma, epsilon: params.epsilon, rows: [] };

	for (let ep=1; ep<=totalEpisodes; ep++){
		if (!running) break;
		if (paused) { await waitWhilePaused(); }

		// reset environment
		world.reset(Number(sizeSelect.value));
		renderGrid(world, gridEl);

		let done = false; let epReward = 0; let steps = 0; let reached = false;

		if (algo === 'qlearning'){
			while(!done && steps < world.maxSteps){
				if (!running) break; if (paused) { await waitWhilePaused(); }
				const s = world.getState();
				const oldRC = world._getState();
				const a = agent.chooseAction(s);
				const res = world.step(a);
				const sNext = res.state;
				epReward += res.reward; steps += 1;
				const newRC = world._getState();
				if (animateEnabled) animateMove(oldRC, newRC, gridEl);
				agent.update(s, a, res.reward, sNext, res.done);
				done = res.done;
				if (world.agent.r === world.goal.r && world.agent.c === world.goal.c) reached = true;
				await sleep(headless ? 0 : 6);
			}
		} else if (algo === 'sarsa'){
			// choose initial action
			let s = world.getState();
			let a = agent.chooseAction(s);
			while(!done && steps < world.maxSteps){
				if (!running) break; if (paused) { await waitWhilePaused(); }
				const oldRC = world._getState();
				const res = world.step(a);
				const sNext = res.state;
				epReward += res.reward; steps += 1;
				const newRC = world._getState();
				if (animateEnabled) animateMove(oldRC, newRC, gridEl);
				const aNext = agent.chooseAction(sNext);
				agent.update(s, a, res.reward, sNext, aNext, res.done);
				s = sNext; a = aNext; done = res.done;
				if (world.agent.r === world.goal.r && world.agent.c === world.goal.c) reached = true;
				await sleep(headless ? 0 : 6);
			}
		} else { // Monte Carlo
			agent.startEpisode();
			let s = world.getState();
			while(!done && steps < world.maxSteps){
				if (!running) break; if (paused) { await waitWhilePaused(); }
				const oldRC = world._getState();
				const a = agent.chooseAction(s);
				const res = world.step(a);
				agent.record(s, a, res.reward);
				epReward += res.reward; steps += 1;
				const newRC = world._getState();
				if (animateEnabled) animateMove(oldRC, newRC, gridEl);
				s = res.state; done = res.done;
				if (world.agent.r === world.goal.r && world.agent.c === world.goal.c) reached = true;
				await sleep(headless ? 0 : 6);
			}
			agent.finishEpisode();
		}

		// update charts per episode
		rewardChart.data.labels.push(ep);
		rewardChart.data.datasets[0].data.push(epReward);
		rewardChart.update();

		stepsChart.data.labels.push(ep);
		stepsChart.data.datasets[0].data.push(steps);
		stepsChart.update();

		rewardsHistory.push(epReward);
		reachedHistory.push(reached ? 1 : 0);
		stepsHistory.push(steps);

		// store row for export
		lastRunResults.rows.push({ episode: ep, reward: epReward, steps, reached: reached ? 1 : 0 });

		// check convergence on recent episodes (window 20)
		if (!converged && checkConvergence(rewardsHistory, reachedHistory, stepsHistory)){
			converged = true;
		}
		const avgWindow = movingAverage(rewardsHistory, 10);
		convChart.data.labels.push(ep);
		convChart.data.datasets[0].data.push(avgWindow);
		convChart.update();

		statusEl.textContent = `Episode ${ep}/${totalEpisodes} — reward ${epReward.toFixed(2)} — steps ${steps} — reached: ${reached} — Converged: ${converged ? 'YES' : 'NO'}`;
		// update policy / heatmap overlays during training (lightweight)
		if (showPolicyCheckbox && showPolicyCheckbox.checked){ try{ renderPolicy(world, gridEl, agent.getQ()); }catch(e){} }
		if (showHeatmapCheckbox && showHeatmapCheckbox.checked){ try{ renderValueHeatmap(world, gridEl, agent.getQ()); }catch(e){} }
		await sleep(8);
	}

	running = false; paused = false;
	if (window._setCurrentAgent) window._setCurrentAgent(null);
	startBtn.disabled = false; pauseBtn.disabled = true; resetBtn.disabled = false; stepBtn.disabled = false; sizeSelect.disabled = false;
}

function stopRun(){ running = false; }

function pauseRun(){ paused = true; pauseBtn.disabled = true; startBtn.disabled = false; }

function resumeRun(){ paused = false; pauseBtn.disabled = false; startBtn.disabled = true; }

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
function waitWhilePaused(){ return new Promise(resolve => {
	const wait = () => { if (!paused) resolve(); else setTimeout(wait, 80); } ; wait();
}); }

function movingAverage(arr, window){ if (arr.length===0) return 0; const start = Math.max(0, arr.length-window); const slice = arr.slice(start); return (slice.reduce((a,b)=>a+b,0)/slice.length).toFixed(2); }

// Simple convergence check over the last `window` episodes.
// Conditions (all must hold):
//  - goal reached at least `minReached` times in the window
//  - reward variation small (max-min <= rewardTol)
//  - steps decreasing: avg of second half lower than first half by at least stepTol
function checkConvergence(rewards, reached, steps, window = 20){
	if (rewards.length < window) return false;
	const start = rewards.length - window;
	const rSlice = rewards.slice(start);
	const reachedSlice = reached.slice(start);
	const stepsSlice = steps.slice(start);

	// 1) goal reached count
	const reachedCount = reachedSlice.reduce((a,b)=>a+b,0);
	const minReached = Math.ceil(0.75 * window); // 15 of 20
	if (reachedCount < minReached) return false;

	// 2) reward stabilized (use range tolerance)
	const rMax = Math.max(...rSlice);
	const rMin = Math.min(...rSlice);
	const rewardTol = 1.0; // tolerance for reward variation
	if ((rMax - rMin) > rewardTol) return false;

	// 3) steps reducing: compare first half vs second half averages
	const half = Math.floor(window/2);
	const firstHalf = stepsSlice.slice(0, half);
	const secondHalf = stepsSlice.slice(half);
	const avg = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
	const avgFirst = avg(firstHalf);
	const avgSecond = avg(secondHalf);
	const stepTol = 0.5; // require at least small decrease
	if ((avgFirst - avgSecond) < stepTol) return false;

	return true;
}

// Run headless comparisons between Q-Learning and SARSA
async function compareAlgorithms({repeats = 5, episodes = 500, size = 5}){
	const algos = ['qlearning','sarsa'];
	const aggregated = {};
	for (const algo of algos){
		aggregated[algo] = { rewardsPerEpisode: Array(episodes).fill(0), stepsPerEpisode: Array(episodes).fill(0), successPerEpisode: Array(episodes).fill(0), convergeEpisodes: [] };
	}

	// run repeats
	for (let run=0; run<repeats; run++){
		for (const algo of algos){
			// instantiate agent
			const states = size * size; const actions = 4;
			const params = { alpha: Number(alphaSlider.value), gamma: Number(gammaSlider.value), epsilon: Number(epsilonSlider.value) };
			let agent = (algo === 'qlearning') ? new QLearningAgent({states, actions, ...params}) : new SARSAAgent({states, actions, ...params});

			// reset world
			world.reset(size);
			// arrays per episode for this run
			const runRewards = [];
			const runSteps = [];
			const runReached = [];

			for (let ep=0; ep<episodes; ep++){
				world.reset(size);
				let done = false; let epReward = 0; let steps = 0; let reached = false;

				if (algo === 'qlearning'){
					while(!done && steps < world.maxSteps){
						const s = world.getState();
						const a = agent.chooseAction(s);
						const res = world.step(a);
						agent.update(s,a,res.reward,res.state,res.done);
						epReward += res.reward; steps += 1; done = res.done;
						if (res.done && world.agent.r===world.goal.r && world.agent.c===world.goal.c) reached = true;
					}
				} else { // SARSA
					let s = world.getState();
					let a = agent.chooseAction(s);
					while(!done && steps < world.maxSteps){
						const res = world.step(a);
						const sNext = res.state;
						const aNext = agent.chooseAction(sNext);
						agent.update(s,a,res.reward,sNext,aNext,res.done);
						s = sNext; a = aNext;
						epReward += res.reward; steps += 1; done = res.done;
						if (res.done && world.agent.r===world.goal.r && world.agent.c===world.goal.c) reached = true;
					}
				}

				runRewards.push(epReward);
				runSteps.push(steps);
				runReached.push(reached ? 1 : 0);
			}

			// accumulate into aggregated arrays
			for (let i=0;i<episodes;i++){
				aggregated[algo].rewardsPerEpisode[i] += runRewards[i];
				aggregated[algo].stepsPerEpisode[i] += runSteps[i];
				aggregated[algo].successPerEpisode[i] += runReached[i];
			}

			// detect convergence episodes for this run (if any)
			let convergeEp = null;
			for (let ep = 20; ep <= episodes; ep++){
				const sliceR = runRewards.slice(0, ep);
				const sliceS = runReached.slice(0, ep);
				const sliceSt = runSteps.slice(0, ep);
				if (checkConvergence(sliceR, sliceS, sliceSt, 20)) { convergeEp = ep; break; }
			}
			aggregated[algo].convergeEpisodes.push(convergeEp === null ? episodes : convergeEp);
		}
	}

	// average aggregated arrays
	for (const algo of algos){
		for (let i=0;i<episodes;i++){
			aggregated[algo].rewardsPerEpisode[i] /= repeats;
			aggregated[algo].stepsPerEpisode[i] /= repeats;
			aggregated[algo].successPerEpisode[i] /= repeats;
		}
		// compute avg converge episode
		const convs = aggregated[algo].convergeEpisodes.filter(x=>x!=null);
		aggregated[algo].avgConverge = convs.length? (convs.reduce((a,b)=>a+b,0)/convs.length) : null;
		// average path quality: mean steps when success in final 100 episodes
		const endWindow = Math.min(100, episodes);
		let sumStepsWhenSuccess = 0, countSuccess=0;
		for (let i=episodes-endWindow;i<episodes;i++){
			// approximate by using successPerEpisode averaged across repeats multiplied by repeats; here we can't reconstruct per-run steps, so we use averaged steps and success rate
			const succRate = aggregated[algo].successPerEpisode[i];
			sumStepsWhenSuccess += aggregated[algo].stepsPerEpisode[i] * succRate;
			countSuccess += succRate;
		}
		aggregated[algo].avgStepsWhenSuccess = countSuccess ? (sumStepsWhenSuccess / countSuccess) : null;
	}

	return aggregated;
}

function renderComparison(aggregated){
	// prepare reward comparison chart
	const episodes = aggregated.qlearning.rewardsPerEpisode.length;
	const labels = Array.from({length:episodes},(_,i)=>i+1);
	if (cmpRewardCtx){
		if (cmpRewardChart) cmpRewardChart.destroy();
		cmpRewardChart = new Chart(cmpRewardCtx, { type:'line', data:{ labels, datasets:[ {label:'Q-Learning', data:aggregated.qlearning.rewardsPerEpisode, borderColor:'#2b8cff', fill:false}, {label:'SARSA', data:aggregated.sarsa.rewardsPerEpisode, borderColor:'#e05555', fill:false} ] }, options:{responsive:true, maintainAspectRatio:false} });
	}
	if (cmpStepsCtx){
		if (cmpStepsChart) cmpStepsChart.destroy();
		cmpStepsChart = new Chart(cmpStepsCtx, { type:'line', data:{ labels, datasets:[ {label:'Q-Learning steps', data:aggregated.qlearning.stepsPerEpisode, borderColor:'#2b8cff', fill:false}, {label:'SARSA steps', data:aggregated.sarsa.stepsPerEpisode, borderColor:'#e05555', fill:false} ] }, options:{responsive:true, maintainAspectRatio:false} });
	}
	if (cmpConvergeCtx){
		if (cmpConvergeChart) cmpConvergeChart.destroy();
		const data = { labels:['Q-Learning','SARSA'], datasets:[{label:'Avg episodes to converge', data:[aggregated.qlearning.avgConverge || 0, aggregated.sarsa.avgConverge || 0], backgroundColor:['#2b8cff','#e05555'] }] };
		cmpConvergeChart = new Chart(cmpConvergeCtx, { type:'bar', data, options:{responsive:true, maintainAspectRatio:false, scales:{y:{beginAtZero:true}}} });
	}
	if (cmpPathCtx){
		if (cmpPathChart) cmpPathChart.destroy();
		const data = { labels:['Q-Learning','SARSA'], datasets:[{label:'Avg steps when successful (last window)', data:[aggregated.qlearning.avgStepsWhenSuccess || 0, aggregated.sarsa.avgStepsWhenSuccess || 0], backgroundColor:['#2b8cff','#e05555'] }] };
		cmpPathChart = new Chart(cmpPathCtx, { type:'bar', data, options:{responsive:true, maintainAspectRatio:false, scales:{y:{beginAtZero:true}}} });
	}

	statusEl.textContent = `Comparison complete.`;
}

// wire UI
sizeSelect.addEventListener('change', ()=>reset());
resetBtn.addEventListener('click', ()=>{ reset(); rewardChart.data.labels=[]; rewardChart.data.datasets[0].data=[]; rewardChart.update(); stepsChart.data.labels=[]; stepsChart.data.datasets[0].data=[]; stepsChart.update(); convChart.data.labels=[]; convChart.data.datasets[0].data=[]; convChart.update(); });
stepBtn.addEventListener('click', ()=>{ const res = world.step(randomAction()); renderGrid(world, gridEl); });
startBtn.addEventListener('click', ()=>{
	if (paused){ resumeRun(); return; }
	const eps = Number(episodesInput.value) || 100; runEpisodes(eps);
});
pauseBtn.addEventListener('click', ()=>{ if (!running) return; pauseRun(); });

// slider value displays
function fmt(v){ return Number(v).toFixed(2); }
if (alphaVal){ alphaVal.textContent = fmt(alphaSlider.value); alphaSlider.addEventListener('input', ()=>{ alphaVal.textContent = fmt(alphaSlider.value); }); }
if (gammaVal){ gammaVal.textContent = fmt(gammaSlider.value); gammaSlider.addEventListener('input', ()=>{ gammaVal.textContent = fmt(gammaSlider.value); }); }
if (epsilonVal){ epsilonVal.textContent = fmt(epsilonSlider.value); epsilonSlider.addEventListener('input', ()=>{ epsilonVal.textContent = fmt(epsilonSlider.value); }); }

exportBtn.addEventListener('click', ()=>{
	if (!lastRunResults || !lastRunResults.rows || lastRunResults.rows.length===0){
		alert('No run data available to export. Run a training session first.');
		return;
	}
	const rows = [];
	rows.push(['algorithm','episodes','alpha','gamma','epsilon','episode','reward','steps','reached'].join(','));
	for (const r of lastRunResults.rows){
		rows.push([lastRunResults.algorithm, lastRunResults.episodes, lastRunResults.alpha, lastRunResults.gamma, lastRunResults.epsilon, r.episode, r.reward.toFixed(4), r.steps, r.reached].join(','));
	}
	const blob = new Blob([rows.join('\n')], {type:'text/csv'});
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = `run-${lastRunResults.algorithm}-${Date.now()}.csv`;
	document.body.appendChild(a);
	a.click();
	a.remove();
	URL.revokeObjectURL(url);
});

// Compare algorithms UI wiring
const compareBtn = document.getElementById('compareBtn');
const compareRepeats = document.getElementById('compareRepeats');
const compareEpisodes = document.getElementById('compareEpisodes');
const compareSize = document.getElementById('compareSize');

compareBtn?.addEventListener('click', async ()=>{
	const repeats = Number(compareRepeats.value) || 5;
	const episodes = Number(compareEpisodes.value) || 500;
	const size = Number(compareSize.value) || 5;
	// run headless comparisons for Q-Learning and SARSA
	compareBtn.disabled = true;
	statusEl.textContent = `Running comparison: ${repeats} repeats, ${episodes} episodes...`;
	const results = await compareAlgorithms({repeats, episodes, size});
	renderComparison(results);
	compareBtn.disabled = false;
});

// initial render
renderGrid(world, gridEl);

// helper to update overlays
function updateOverlays(){
	try{
		const agent = window.currentAgent;
		if (showHeatmapCheckbox && showHeatmapCheckbox.checked && agent && agent.getQ) renderValueHeatmap(world, gridEl, agent.getQ());
		else gridEl.querySelectorAll('.heat').forEach(e=>e.remove());
		if (showPolicyCheckbox && showPolicyCheckbox.checked && agent && agent.getQ) renderPolicy(world, gridEl, agent.getQ());
		else gridEl.querySelectorAll('.policy').forEach(e=>e.remove());
	}catch(e){ console.warn('updateOverlays', e); }
}

// listen for toggles
showPolicyCheckbox?.addEventListener('change', ()=>{ updateOverlays(); });
showHeatmapCheckbox?.addEventListener('change', ()=>{ updateOverlays(); });

// Maze edit: click cells to toggle obstacles; Shift+click sets start, Alt+click sets goal
gridEl.addEventListener('click', (ev)=>{
	if (!editMazeCheckbox || !editMazeCheckbox.checked) return;
	const cell = ev.target.closest('.cell'); if (!cell) return;
	const r = Number(cell.dataset.r); const c = Number(cell.dataset.c);
	if (ev.shiftKey){ // set start
		world.start = {r,c}; world.agent = {r,c};
	} else if (ev.altKey){ // set goal
		world.goal = {r,c};
	} else {
		// toggle obstacle
		world.grid[r][c] = world.grid[r][c] === 1 ? 0 : 1;
		// ensure start/goal not on obstacle
		if (world.grid[world.start.r][world.start.c] === 1) world.grid[world.start.r][world.start.c] = 0;
		if (world.grid[world.goal.r][world.goal.c] === 1) world.grid[world.goal.r][world.goal.c] = 0;
	}
	renderGrid(world, gridEl);
	updateOverlays();
});

// expose a hook for training to set currentAgent so overlays can use it
window._setCurrentAgent = a => { window.currentAgent = a; updateOverlays(); };
