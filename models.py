import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Action(BaseModel):
    missing_values: List[Dict[str, Any]] = Field(default_factory=list, description="List of dicts with 'row_idx' and 'col_name'")
    type_errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of dicts with 'row_idx' and 'col_name'")
    duplicates: List[int] = Field(default_factory=list, description="List of row_idx that are duplicates of previous rows")
    fix_suggestions: List[Dict[str, Any]] = Field(default_factory=list, description="List of dicts with 'row_idx', 'col_name', 'suggested_value'")
    done: bool = Field(False, description="Set this to true when the audit is complete and want to submit for grading.")

class Observation(BaseModel):
    dataset: List[Dict[str, Any]] = Field(..., description="The dataset as a list of dictionaries.")
    dataset_schema: Dict[str, str] = Field(..., description="Expected data types for each column.")
    task_level: str = Field(..., description="'easy', 'medium', or 'hard'")
    instruction: str = Field(..., description="Instructions for the agent.")

class Reward(BaseModel):
    reward: float
    done: bool
    info: Dict[str, Any]

class DataAuditEnv:
    def __init__(self):
        self.state_data = None
        self.ground_truth = None
        self.current_reward = 0.0
        self.is_done = False
        self.task_level = "easy"
        self.max_steps = 10
        self.steps = 0
        
    def _generate_dataset(self, level: str):
        # Generate clean data
        N = 20
        clean_data = pd.DataFrame({
            "id": range(1, N+1),
            "name": [f"Person_{i}" for i in range(N)],
            "age": np.random.randint(18, 65, size=N),
            "salary": np.random.randint(40000, 120000, size=N).astype(float)
        })
        expected_schema = {
            "id": "int",
            "name": "string",
            "age": "int",
            "salary": "float"
        }
        
        dirty_data = clean_data.copy()
        dirty_data = dirty_data.astype(object)
        ground_truth = {
            "missing": [],
            "type_errors": [],
            "duplicates": [],
            "fixes": []
        }
        
        if level in ["easy", "hard"]:
            # Inject missing values
            indices = np.random.choice(N, size=3, replace=False)
            for idx in indices:
                col = np.random.choice(["age", "salary"])
                dirty_data.at[idx, col] = np.nan
                ground_truth["missing"].append({"row_idx": int(idx), "col_name": col})
                ground_truth["fixes"].append({"row_idx": int(idx), "col_name": col, "suggested_value": "missing"})
                
        if level in ["medium", "hard"]:
            # Inject type errors
            indices = np.random.choice(N, size=3, replace=False)
            for idx in indices:
                if not any(m["row_idx"] == idx for m in ground_truth["missing"]):
                    col = "age"
                    old_val = dirty_data.at[idx, col]
                    dirty_data.at[idx, col] = str(old_val) + " years"
                    ground_truth["type_errors"].append({"row_idx": int(idx), "col_name": col})
                    
            # Inject duplicates
            dup_idx_source = np.random.choice(N, size=2, replace=False)
            for src in dup_idx_source:
                new_idx = len(dirty_data)
                dirty_data.loc[new_idx] = dirty_data.iloc[src]
                ground_truth["duplicates"].append(new_idx)

        # Convert to list of dicts. Replace nan with None
        dirty_data = dirty_data.replace({np.nan: None})
        
        return dirty_data.to_dict(orient="records"), expected_schema, ground_truth

    def reset(self, task_level: str = "easy") -> Observation:
        print("[START]")
        self.task_level = task_level
        dataset, schema, gt = self._generate_dataset(task_level)
        self.state_data = dataset
        self.ground_truth = gt
        self.is_done = False
        self.current_reward = 0.0
        self.steps = 0
        
        instructions = {
            "easy": "Identify all missing values in the dataset. Return a list of dicts with 'row_idx' and 'col_name'.",
            "medium": "Identify type errors (e.g. string where int expected) and duplicates. Provide 'row_idx' and 'col_name' for type errors, and a list of 'row_idx' for duplicates.",
            "hard": "Perform a full audit: find missing values, type errors, duplicates, and provide fix suggestions for missing values."
        }
        
        obs = Observation(
            dataset=self.state_data,
            dataset_schema=schema,
            task_level=self.task_level,
            instruction=instructions.get(task_level, "")
        )
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward]:
        if self.is_done:
            info = {"error": "Environment is already done."}
            return self.state(), Reward(reward=0.0, done=True, info=info)
            
        print("[STEP]")
        self.steps += 1
        
        if action.done or self.steps >= self.max_steps:
            self.is_done = True
            
        reward, grading_info = self._grade(action)
        self.current_reward = reward
        
        if self.is_done:
            print("[END]")
            
        return self.state(), Reward(reward=reward, done=self.is_done, info=grading_info)
        
    def _grade(self, action: Action) -> tuple[float, dict]:
        score = 0.0
        info = {}
        
        def dict_list_match_score(pred_list, true_list, required_keys):
            if not true_list: return 1.0 if not pred_list else 0.0
            matches = 0
            for t in true_list:
                for p in pred_list:
                    if all(p.get(k) == t.get(k) for k in required_keys):
                        matches += 1
                        break
            precision = matches / max(len(pred_list), 1)
            recall = matches / len(true_list)
            return (precision + recall) / 2
            
        if self.task_level == "easy":
            score = dict_list_match_score(action.missing_values, self.ground_truth["missing"], ["row_idx", "col_name"])
            info["missing_score"] = score
            
        elif self.task_level == "medium":
            type_score = dict_list_match_score(action.type_errors, self.ground_truth["type_errors"], ["row_idx", "col_name"])
            
            pred_dups = set(action.duplicates)
            true_dups = set(self.ground_truth["duplicates"])
            if not true_dups:
                dup_score = 1.0 if not pred_dups else 0.0
            else:
                tp = len(pred_dups.intersection(true_dups))
                prec = tp / max(len(pred_dups), 1)
                rec = tp / len(true_dups)
                dup_score = (prec + rec) / 2
                
            score = (type_score + dup_score) / 2
            info["type_score"] = type_score
            info["dup_score"] = dup_score
            
        elif self.task_level == "hard":
            missing_score = dict_list_match_score(action.missing_values, self.ground_truth["missing"], ["row_idx", "col_name"])
            type_score = dict_list_match_score(action.type_errors, self.ground_truth["type_errors"], ["row_idx", "col_name"])
            
            pred_dups = set(action.duplicates)
            true_dups = set(self.ground_truth["duplicates"])
            if not true_dups:
                dup_score = 1.0 if not pred_dups else 0.0
            else:
                tp = len(pred_dups.intersection(true_dups))
                prec = tp / max(len(pred_dups), 1)
                rec = tp / len(true_dups)
                dup_score = (prec + rec) / 2
            
            score = (missing_score + type_score + dup_score) / 3
            info["components"] = {"missing": missing_score, "type": type_score, "dup": dup_score}
            
        return score, info

    def state(self) -> Observation:
        schema = {
            "id": "int",
            "name": "string",
            "age": "int",
            "salary": "float"
        }
        instructions = {
            "easy": "Identify all missing values in the dataset. Return a list of dicts with 'row_idx' and 'col_name'.",
            "medium": "Identify type errors (e.g. string where int expected) and duplicates. Provide 'row_idx' and 'col_name' for type errors, and a list of 'row_idx' for duplicates.",
            "hard": "Perform a full audit: find missing values, type errors, duplicates, and provide fix suggestions for missing values."
        }
        return Observation(
            dataset=self.state_data,
            dataset_schema=schema,
            task_level=self.task_level,
            instruction=instructions.get(self.task_level, "")
        )
