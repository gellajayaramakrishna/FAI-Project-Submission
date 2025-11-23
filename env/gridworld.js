// Simple GridWorld environment
export class GridWorld {
	constructor(size = 5, maxSteps = 200) {
		this.size = size;
		this.maxSteps = maxSteps;
		this.stepCost = -0.1;
		this.goalReward = 10;
		this.obstaclePenalty = -1;
		this.reset();
	}

	// preset layout: start at top-left, goal at bottom-right, some obstacles
	_defaultLayout() {
		const n = this.size;
		const grid = Array(n).fill(null).map(()=>Array(n).fill(0));
		// 0 empty, 1 obstacle
		if (n === 5) {
			grid[1][2] = 1;
			grid[2][2] = 1;
			grid[3][1] = 1;
		} else {
			grid[1][3] = 1;
			grid[2][3] = 1;
			grid[3][1] = 1;
			grid[4][4] = 1;
		}
		return grid;
	}

	reset(size) {
		if (size) this.size = size;
		this.grid = this._defaultLayout();
		this.start = {r:0,c:0};
		this.goal = {r:this.size-1,c:this.size-1};
		this.agent = {r:this.start.r,c:this.start.c};
		this.stepCount = 0;
		this.done = false;
		return this.getState();
	}

	// Public accessor for current agent state as an integer:
	// state = row * cols + col
	getState(){
		return this.agent.r * this.size + this.agent.c;
	}

	// Convert integer state -> {r,c}
	stateToRC(state){
		const r = Math.floor(state / this.size);
		const c = state % this.size;
		return {r,c};
	}

	// Backwards-compatible accessor that returns {r,c}
	_getState(){
		return {r:this.agent.r, c:this.agent.c};
	}

	// Returns whether the current episode is terminal
	isTerminal(){
		return !!this.done;
	}

	inBounds(r,c){
		return r>=0 && c>=0 && r<this.size && c<this.size;
	}

	isObstacle(r,c){
		return this.grid[r][c] === 1;
	}

	// action: 0=up,1=right,2=down,3=left  OR 'up'/'right'... string
	step(action) {
		if (this.done) return {state:this._getState(), reward:0, done:true};
		this.stepCount += 1;
		const dirs = {0:[-1,0],1:[0,1],2:[1,0],3:[0,-1], up:[-1,0], right:[0,1], down:[1,0], left:[0,-1]};
		const d = dirs[action];
		if (!d) throw new Error('Invalid action');
		const nr = this.agent.r + d[0];
		const nc = this.agent.c + d[1];
		let reward = this.stepCost; // step cost

		if (!this.inBounds(nr,nc)) {
			// out-of-bounds: apply penalty and agent stays in place
			reward += this.obstaclePenalty;
			this.agent = {r:this.agent.r,c:this.agent.c};
		} else if (this.isObstacle(nr,nc)) {
			// obstacle penalty, agent stays in place
			reward += this.obstaclePenalty;
		} else {
			// valid move
			this.agent = {r:nr,c:nc};
		}

		// check goal
		if (this.agent.r === this.goal.r && this.agent.c === this.goal.c) {
			reward += this.goalReward;
			this.done = true;
		}

		if (this.stepCount >= this.maxSteps) this.done = true;

		return {state:this.getState(), reward, done:this.done};
	}
}
