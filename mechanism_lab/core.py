from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

@dataclass(frozen=True)
class Task:
    trials: int = 160
    reversal: int = 80
    high: float = .8
    low: float = .2
    seed: int = 42

    def probabilities(self, t):
        return (self.high, self.low) if t < self.reversal else (self.low, self.high)

class Agent:
    name = 'base'
    bounds = ()
    initial = ()
    def __init__(self, *params):
        self.params = params or self.initial
        self.reset()
    def reset(self):
        self.values = np.array([.5, .5], dtype=float)
    def probability(self):
        raise NotImplementedError
    def update(self, choice, reward):
        pass

class RandomAgent(Agent):
    name = 'random'
    def probability(self):
        return .5

class QLearning(Agent):
    name = 'qlearning'
    bounds = ((.001, .999), (.05, 20.))
    initial = (.2, 5.)
    def probability(self):
        return float(expit(self.params[1] * (self.values[1] - self.values[0])))
    def update(self, choice, reward):
        self.values[choice] += self.params[0] * (reward - self.values[choice])

class AdaptiveLearning(Agent):
    """Prediction-error-dependent learning rate; a descriptive heuristic, not a Bayesian model."""
    name = 'adaptive'
    bounds = ((.001, .95), (0., .95), (.05, 20.))
    initial = (.1, .5, 5.)
    def probability(self):
        return float(expit(self.params[2] * (self.values[1] - self.values[0])))
    def update(self, choice, reward):
        error = reward - self.values[choice]
        rate = min(.999, self.params[0] + self.params[1] * abs(error))
        self.values[choice] += rate * error

class LeakyIntegrator(Agent):
    """Two competing, leaky activity traces with a softmax readout."""
    name = 'leaky'
    bounds = ((.001, .999), (.001, 1.), (.05, 20.))
    initial = (.12, .3, 5.)
    def probability(self):
        return float(expit(self.params[2] * (self.values[1] - self.values[0])))
    def update(self, choice, reward):
        leak, drive, _ = self.params
        self.values = (1 - leak) * self.values + leak * .5
        self.values[choice] += drive * (reward - self.values[choice])

MODELS = {cls.name: cls for cls in (RandomAgent, QLearning, AdaptiveLearning, LeakyIntegrator)}

def simulate(task, model='qlearning', params=None, seed=None):
    rng = np.random.default_rng(task.seed if seed is None else seed)
    agent = MODELS[model](*(params or ()))
    rows = []
    for t in range(task.trials):
        p1 = float(np.clip(agent.probability(), 1e-8, 1-1e-8))
        choice = int(rng.random() < p1)
        probs = task.probabilities(t)
        reward = int(rng.random() < probs[choice])
        rows.append({'trial':t+1, 'choice':choice, 'reward':reward, 'p_choice1':p1,
                     'optimal':int(choice == int(probs[1] > probs[0])),
                     'values':agent.values.tolist()})
        agent.update(choice, reward)
    return rows

def log_likelihood(model, params, observations):
    agent = MODELS[model](*params)
    ll = 0.
    for row in observations:
        p = float(np.clip(agent.probability(), 1e-9, 1-1e-9))
        ll += np.log(p if row['choice'] == 1 else 1-p)
        agent.update(row['choice'], row['reward'])
    return float(ll)

def fit(model, observations, restarts=5, seed=123):
    cls = MODELS[model]
    if not cls.bounds:
        ll = log_likelihood(model, (), observations)
        return {'model':model, 'params':[], 'log_likelihood':ll, 'aic':-2*ll, 'bic':-2*ll}
    rng = np.random.default_rng(seed)
    best = None
    starts = [cls.initial] + [tuple(rng.uniform(lo, hi) for lo,hi in cls.bounds) for _ in range(restarts-1)]
    for start in starts:
        result = minimize(lambda p: -log_likelihood(model, p, observations), start,
                          bounds=cls.bounds, method='L-BFGS-B')
        if best is None or result.fun < best.fun:
            best = result
    k = len(cls.bounds)
    ll = -float(best.fun)
    return {'model':model, 'params':[float(x) for x in best.x],
            'log_likelihood':ll, 'aic':2*k-2*ll,
            'bic':k*np.log(len(observations))-2*ll}

def compare(observations):
    return sorted((fit(name, observations) for name in MODELS), key=lambda x:x['bic'])

def recover(model='qlearning', params=None, repeats=12, task=None):
    task = task or Task()
    true_params = params or MODELS[model].initial
    estimates = []
    winners = {name:0 for name in MODELS}
    for seed in range(repeats):
        rows = simulate(task, model, true_params, seed=seed+1000)
        rankings = compare(rows)
        winners[rankings[0]['model']] += 1
        estimates.append(fit(model, rows)['params'])
    return {'generating_model':model, 'true_params':list(true_params),
            'estimates':estimates, 'winner_counts':winners}

def intervention(task, model='qlearning', baseline=None, changed=None, repetitions=30):
    baseline = baseline or MODELS[model].initial
    changed = changed or baseline
    def run(params):
        runs = [simulate(task, model, params, seed=5000+i) for i in range(repetitions)]
        return {'optimal_rate':float(np.mean([np.mean([r['optimal'] for r in run]) for run in runs])),
                'reward_rate':float(np.mean([np.mean([r['reward'] for r in run]) for run in runs])),
                'post_reversal_optimal':float(np.mean([np.mean([r['optimal'] for r in run[task.reversal:]]) for run in runs]))}
    return {'baseline':run(baseline), 'intervention':run(changed), 'repetitions':repetitions,
            'note':'Matched random seeds reduce simulation noise; these are model interventions, not biological predictions.'}
