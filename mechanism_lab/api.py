from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .core import Task, MODELS, simulate, compare, recover, intervention

app = FastAPI(title='Mechanism Lab', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET','POST'], allow_headers=['*'])
class Experiment(BaseModel):
    model: str = 'qlearning'
    trials: int = Field(default=160, ge=20, le=1000)
    reversal: int = Field(default=80, ge=1)
    seed: int = 42
    params: list[float] | None = None

def unpack(req):
    if req.model not in MODELS: raise HTTPException(400, 'Unknown model')
    if req.reversal >= req.trials: raise HTTPException(400, 'Reversal must precede final trial')
    if req.params is not None:
        bounds = MODELS[req.model].bounds
        if len(req.params) != len(bounds) or any(not lo <= v <= hi for v,(lo,hi) in zip(req.params,bounds)):
            raise HTTPException(400, 'Parameters outside model bounds')
    return Task(req.trials, req.reversal, seed=req.seed)

@app.get('/models')
def models():
    return {name:{'defaults':cls.initial,'bounds':cls.bounds} for name,cls in MODELS.items()}
@app.post('/simulate')
def simulation(req:Experiment):
    return simulate(unpack(req), req.model, req.params)
@app.post('/compare')
def comparison(req:Experiment):
    return compare(simulate(unpack(req), req.model, req.params))
@app.post('/recover')
def recovery(req:Experiment):
    return recover(req.model, req.params, repeats=8, task=unpack(req))
@app.post('/intervene')
def intervention_endpoint(req:Experiment):
    task = unpack(req)
    return intervention(task, req.model, changed=req.params)
