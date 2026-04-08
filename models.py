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
        self.found_ground_truth = set() # For partial progress tracking
        
    def _generate_dataset(self, level: str):
        # Generate clean data
        N = 50 if level == "hard" else 20
        clean_data = pd.DataFrame({
            "id": range(1, N+1),
            "name": [f"Person_{i}" for i in range(N)],
            "age": np.random.randint(18, 65, size=N),
            "salary": np.random.randint(40000, 120000, size=N).astype(float),
            "joined_date": pd.date_range(start="2020-01-01", periods=N, freq="D").strftime("%Y-%m-%d")
        })
        expected_schema = {
            "id": "int",
            "name": "string",
            "age": "int",
            "salary": "float",
            "joined_date": "date (%Y-%m-%d)"
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
            indices = np.random.choice(N, size=3 if level=="easy" else 5, replace=False)
            for idx in indices:
                col = np.random.choice(["age", "salary"])
                dirty_data.at[idx, col] = np.nan
                ground_truth["missing"].append({"row_idx": int(idx), "col_name": col})
                # Sensible fix: median for age, mean for salary
                fix_val = 30 if col == "age" else 60000.0
                ground_truth["fixes"].append({"row_idx": int(idx), "col_name": col, "suggested_value": fix_val})
                
        if level in ["medium", "hard"]:
            # Inject type errors
            num_errors = 3 if level=="medium" else 6
            indices = np.random.choice(N, size=num_errors, replace=False)
            for idx in indices:
                # Avoid overlap with missing
                if not any(m["row_idx"] == idx for m in ground_truth["missing"]):
                    error_type = np.random.choice(["string_int", "date_format"]) if level=="hard" else "string_int"
                    
                    if error_type == "string_int":
                        col = "age"
                        old_val = dirty_data.at[idx, col]
                        dirty_data.at[idx, col] = str(old_val) + " years"
                        ground_truth["type_errors"].append({"row_idx": int(idx), "col_name": col})
                        ground_truth["fixes"].append({"row_idx": int(idx), "col_name": col, "suggested_value": int(old_val)})
                    else:
                        col = "joined_date"
                        old_val = dirty_data.at[idx, col]
                        # Change format to MM/DD/YYYY
                        dt = pd.to_datetime(old_val)
                        dirty_data.at[idx, col] = dt.strftime("%m/%d/%Y")
                        ground_truth["type_errors"].append({"row_idx": int(idx), "col_name": col})
                        ground_truth["fixes"].append({"row_idx": int(idx), "col_name": col, "suggested_value": old_val})
            
            # Inject outliers for hard
            if level == "hard":
                outlier_idx = np.random.choice(N)
                dirty_data.at[outlier_idx, "salary"] = 1000000.0 # Clear outlier
                ground_truth["type_errors"].append({"row_idx": int(outlier_idx), "col_name": "salary"}) # treated as anomaly
                ground_truth["fixes"].append({"row_idx": int(outlier_idx), "col_name": "salary", "suggested_value": 80000.0})

            # Inject duplicates
            num_dups = 2 if level=="medium" else 4
            dup_idx_source = np.random.choice(N, size=num_dups, replace=False)
            for src in dup_idx_source:
                new_idx = len(dirty_data)
                row_data = dirty_data.iloc[src].copy()
                if level == "hard" and np.random.random() > 0.5:
                    # Subtle duplicate: slight change in name
                    row_data["name"] = row_data["name"] + " "
                dirty_data.loc[new_idx] = row_data
                ground_truth["duplicates"].append(new_idx)

        # Convert to list of dicts. Replace nan with None
        dirty_data = dirty_data.replace({np.nan: None})
        
        return dirty_data.to_dict(orient="records"), expected_schema, ground_truth

    def reset(self, task_level: str = "easy") -> Observation:
        self.task_level = task_level
        dataset, schema, gt = self._generate_dataset(task_level)
        self.state_data = dataset
        self.ground_truth = gt
        self.is_done = False
        self.current_reward = 0.0
        self.steps = 0
        self.found_ground_truth = set()
        
        instructions = {
            "easy": "Identify all missing values in the dataset. Return a list of dicts with 'row_idx' and 'col_name'.",
            "medium": "Identify type errors (e.g. string where int expected) and duplicates. Provide 'row_idx' and 'col_name' for type errors, and a list of 'row_idx' for duplicates.",
            "hard": "Perform a full audit: find missing values, type errors (including date format and outliers), duplicates, and provide fix suggestions for all identified issues."
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
            
        self.steps += 1
        
        if action.done or self.steps >= self.max_steps:
            self.is_done = True
            
        reward, grading_info = self._grade(action)
        self.current_reward = reward
        
        return self.state(), Reward(reward=reward, done=self.is_done, info=grading_info)
        
    def _grade(self, action: Action) -> tuple[float, dict]:
        info = {}
        
        def dict_list_match_score(pred_list, true_list, required_keys, category):
            if not true_list: return 1.0 if not pred_list else 0.0
            matches = 0
            new_matches = 0
            for i, t in enumerate(true_list):
                found = False
                for p in pred_list:
                    if all(p.get(k) == t.get(k) for k in required_keys):
                        found = True
                        break
                if found:
                    matches += 1
                    key = f"{category}_{i}"
                    if key not in self.found_ground_truth:
                        new_matches += 1
                        self.found_ground_truth.add(key)
            
            # Simple average of precision and recall for the step score
            precision = matches / max(len(pred_list), 1)
            recall = matches / len(true_list)
            return (precision + recall) / 2

        if self.task_level == "easy":
            score = dict_list_match_score(action.missing_values, self.ground_truth["missing"], ["row_idx", "col_name"], "missing")
            info["missing_score"] = score
            
        elif self.task_level == "medium":
            type_score = dict_list_match_score(action.type_errors, self.ground_truth["type_errors"], ["row_idx", "col_name"], "type")
            
            pred_dups = set(action.duplicates)
            true_dups = set(self.ground_truth["duplicates"])
            matches = 0
            if not true_dups:
                dup_score = 1.0 if not pred_dups else 0.0
            else:
                tp_set = pred_dups.intersection(true_dups)
                matches = len(tp_set)
                for d_idx in tp_set:
                    self.found_ground_truth.add(f"dup_{d_idx}")
                prec = matches / max(len(pred_dups), 1)
                rec = matches / len(true_dups)
                dup_score = (prec + rec) / 2
                
            score = (type_score + dup_score) / 2
            info["type_score"] = type_score
            info["dup_score"] = dup_score
            
        elif self.task_level == "hard":
            missing_score = dict_list_match_score(action.missing_values, self.ground_truth["missing"], ["row_idx", "col_name"], "missing")
            type_score = dict_list_match_score(action.type_errors, self.ground_truth["type_errors"], ["row_idx", "col_name"], "type")
            
            pred_dups = set(action.duplicates)
            true_dups = set(self.ground_truth["duplicates"])
            if not true_dups:
                dup_score = 1.0 if not pred_dups else 0.0
            else:
                tp_set = pred_dups.intersection(true_dups)
                for d_idx in tp_set:
                    self.found_ground_truth.add(f"dup_{d_idx}")
                prec = len(tp_set) / max(len(pred_dups), 1)
                rec = len(tp_set) / len(true_dups)
                dup_score = (prec + rec) / 2
            
            fix_score = dict_list_match_score(action.fix_suggestions, self.ground_truth["fixes"], ["row_idx", "col_name", "suggested_value"], "fix")
            
            score = (missing_score + type_score + dup_score + fix_score) / 4
            info["components"] = {
                "missing": missing_score, 
                "type": type_score, 
                "dup": dup_score,
                "fix": fix_score
            }
            
        return score, info

    def state(self) -> Observation:
        schema = {
            "id": "int",
            "name": "string",
            "age": "int",
            "salary": "float",
            "joined_date": "date (%Y-%m-%d)"
        }
        instructions = {
            "easy": "Identify all missing values in the dataset. Return a list of dicts with 'row_idx' and 'col_name'.",
            "medium": "Identify type errors (e.g. string where int expected) and duplicates. Provide 'row_idx' and 'col_name' for type errors, and a list of 'row_idx' for duplicates.",
            "hard": "Perform a full audit: find missing values, type errors (including date format and outliers), duplicates, and provide fix suggestions for all identified issues."
        }
        return Observation(
            dataset=self.state_data,
            dataset_schema=schema,
            task_level=self.task_level,
            instruction=instructions.get(self.task_level, "")
        )
