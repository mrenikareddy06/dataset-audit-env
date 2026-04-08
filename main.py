from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models import DataAuditEnv, Action, Observation, Reward

app = FastAPI(title="Dataset Audit OpenEnv")
env = DataAuditEnv()

@app.get("/")
@app.get("/web")
@app.get("/web/")
def get_root():
    return {
        "name": "dataset-audit-env",
        "description": "OpenEnv RL environment for data quality auditing",
        "endpoints": ["/reset", "/step", "/state"],
        "tasks": ["easy", "medium", "hard"],
        "version": "1.0.0"
    }

class ResetRequest(BaseModel):
    task_level: str = "easy"

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = None):
    task_level = req.task_level if req else "easy"
    return env.reset(task_level)

@app.post("/step", response_model=dict)
def step_env(action: Action):
    obs, reward = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump()
    }

@app.get("/state", response_model=Observation)
def get_state():
    if env.state_data is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()
