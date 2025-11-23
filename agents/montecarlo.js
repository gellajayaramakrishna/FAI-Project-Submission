// First-visit Monte Carlo control agent
// API:
//  new MonteCarloAgent({states, actions, alpha, gamma, epsilon})
//  init(states, actions)
//  startEpisode()
//  record(s,a,r)
//  finishEpisode()

export default class MonteCarloAgent {
  constructor({states = 25, actions = 4, alpha = 0.1, gamma = 0.99, epsilon = 0.1} = {}){
    this.states = states;
    this.actions = actions;
    this.alpha = alpha;
    this.gamma = gamma;
    this.epsilon = epsilon;
    this._initQ();
    this.resetEpisode();
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

  // epsilon-greedy
  chooseAction(s){
    this.ensureState(s);
    if (Math.random() < this.epsilon) return Math.floor(Math.random()*this.actions);
    const q = this.Q[s]; let max = -Infinity; const best = [];
    for (let a=0;a<q.length;a++){
      if (q[a] > max){ max = q[a]; best.length = 0; best.push(a); }
      else if (q[a] === max) best.push(a);
    }
    return best[Math.floor(Math.random()*best.length)];
  }

  resetEpisode(){
    this.trajectory = []; // array of {s,a,r}
  }

  startEpisode(){ this.resetEpisode(); }

  record(s,a,r){ this.trajectory.push({s,a,r}); }

  // first-visit returns and updates Q
  finishEpisode(){
    const Gs = new Map(); // map key `${s}|${a}` -> first index
    // accumulate returns backwards
    let G = 0;
    for (let t = this.trajectory.length-1; t>=0; t--){
      const {s,a,r} = this.trajectory[t];
      G = r + this.gamma * G;
      const key = `${s}|${a}`;
      if (!Gs.has(key)){
        // first visit of (s,a) when scanning backwards
        Gs.set(key, true);
        this.ensureState(s);
        const qsa = this.Q[s][a];
        this.Q[s][a] = qsa + this.alpha * (G - qsa);
      }
    }
    this.resetEpisode();
  }

  setParams({alpha, gamma, epsilon} = {}){
    if (alpha !== undefined) this.alpha = alpha;
    if (gamma !== undefined) this.gamma = gamma;
    if (epsilon !== undefined) this.epsilon = epsilon;
  }

  getQ(){ return this.Q; }
}
