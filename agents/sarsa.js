// SARSA agent (on-policy)
// API similar to QLearningAgent

export default class SARSAAgent {
  constructor({states = 25, actions = 4, alpha = 0.1, gamma = 0.99, epsilon = 0.1} = {}){
    this.states = states;
    this.actions = actions;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this._initQ();
  }

  _initQ(){
    this.Q = Array(this.states).fill(null).map(()=>Array(this.actions).fill(0));
  }

  init(states, actions){
    if (states) this.states = states;
    if (actions) this.actions = actions;
    this._initQ();
  }

  ensureState(s){
    if (s >= this.Q.length){
      const add = s - this.Q.length + 1;
      for (let i=0;i<add;i++) this.Q.push(Array(this.actions).fill(0));
    }
  }

  chooseAction(s){
    this.ensureState(s);
    if (Math.random() < this.epsilon){
      return Math.floor(Math.random()*this.actions);
    }
    const q = this.Q[s];
    let max = -Infinity; const best = [];
    for (let a=0;a<q.length;a++){
      if (q[a] > max){ max = q[a]; best.length = 0; best.push(a); }
      else if (q[a] === max) best.push(a);
    }
    return best[Math.floor(Math.random()*best.length)];
  }

  // SARSA update: uses the actually selected next action aNext
  update(s, a, r, sNext, aNext, done=false){
    this.ensureState(s);
    this.ensureState(sNext);
    const qsa = this.Q[s][a];
    const qNext = done ? 0 : this.Q[sNext][aNext];
    const target = r + this.gamma * qNext;
    this.Q[s][a] = qsa + this.alpha * (target - qsa);
  }

  setParams({alpha, gamma, epsilon} = {}){
    if (alpha !== undefined) this.alpha = alpha;
    if (gamma !== undefined) this.gamma = gamma;
    if (epsilon !== undefined) this.epsilon = epsilon;
  }

  getQ(){ return this.Q; }
}
