import os
import json
import asyncio
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

class AsyncEnvWrapper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.http = httpx.AsyncClient(timeout=30.0)
        
    async def reset(self, task_level: str):
        res = await self.http.post(f"{self.base_url}/reset", json={"task_level": task_level})
        res.raise_for_status()
        return res.json()
        
    async def step(self, action: dict):
        res = await self.http.post(f"{self.base_url}/step", json=action)
        res.raise_for_status()
        return res.json()
        
    async def close(self):
        await self.http.aclose()

def build_prompt(obs: dict) -> str:
    prompt = f"Instruction: {obs['instruction']}\n\n"
    prompt += f"Schema: {json.dumps(obs['dataset_schema'])}\n\n"
    prompt += f"Dataset:\n{json.dumps(obs['dataset'], indent=2)}\n\n"
    prompt += """You are an advanced data quality agent.
Your final output must be exactly a JSON object matching this schema:
{
  "missing_values": [{"row_idx": 0, "col_name": "age"}],
  "type_errors": [{"row_idx": 1, "col_name": "salary"}],
  "duplicates": [2],
  "fix_suggestions": [{"row_idx": 0, "col_name": "age", "suggested_value": 30}],
  "done": true
}
Do not write anything other than the JSON object.
"""
    return prompt

async def main():
    env = AsyncEnvWrapper(ENV_URL)
    final_scores = {}
    
    for task_level in ["easy", "medium", "hard"]:
        print(f"[START] task={task_level} env=dataset-audit-env model={MODEL_NAME}")
        
        success = "false"
        step = 0
        rewards = []
        score = 0.00
        
        try:
            obs = await env.reset(task_level=task_level)
            done = False
            
            while not done and step < MAX_STEPS:
                prompt = build_prompt(obs)
                error = "null"
                action_data = None
                action_log_str = "{}"
                
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a data auditing agent."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    
                    action_str = response.choices[0].message.content.strip()
                    if "```json" in action_str:
                        action_str = action_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in action_str:
                        action_str = action_str.split("```")[1].split("```")[0].strip()
                    
                    action_data = json.loads(action_str)
                    action_data["done"] = True
                    action_log_str = json.dumps(action_data)
                except Exception as e:
                    error = str(e)
                    action_data = {
                        "missing_values": [],
                        "type_errors": [],
                        "duplicates": [],
                        "fix_suggestions": [],
                        "done": True
                    }
                    action_log_str = "{}"
                    
                step += 1
                
                try:
                    payload = await env.step(action_data)
                    obs = payload["observation"]
                    curr_reward = payload["reward"]
                    done = payload["done"]
                except Exception as e:
                    error = str(e) if error == "null" else f"{error} | {e}"
                    done = True
                    curr_reward = 0.0
                    
                rewards.append(curr_reward)
                done_str = "true" if done else "false"
                err_val = f"'{error}'" if error != "null" else "null"
                print(f"[STEP] step={step} action='{action_log_str}' reward={curr_reward:.2f} done={done_str} error={err_val}")
                
            success = "true" if done else "false"
            score = rewards[-1] if rewards else 0.00
            
        except Exception as e:
            print(f"Critical error in task {task_level}: {e}")
        finally:
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(f"[END] success={success} steps={step} score={score:.2f} rewards={rewards_str}")
            final_scores[task_level] = score
            
    await env.close()
    
    print("\n[SUMMARY]")
    for lvl, s in final_scores.items():
        print(f"Task: {lvl} | Score: {s:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
