---
title: Dataset Audit Env
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - data-auditing
---
# Dataset Audit Environment

An OpenEnv-compliant reinforcement learning environment engineered for rigorous data quality testing against modern LLM agents.

---

## 1. Environment Description and Motivation

In the modern data engineering lifecycle, maintaining data integrity across downstream platforms remains a critical and highly repetitive constraint. Datasets frequently harbor structural deficiencies—hidden missing values, mismatched schema data types, and logical duplicates—that can easily compromise the performance of machine learning applications.

The **Dataset Audit Environment** simulates an automated, real-world data engineering and analytical quality control task. It objectively measures the capacity of autonomous AI agents to ingest a "dirty" tabulated dataset, isolate structural anomalies programmatically, and actively deploy error resolution techniques. This benchmark requires autonomous agents to leverage exact, structural reasoning as opposed to open-ended text modeling.

---

## 2. Action Space Definition

Agents interact exclusively using a strictly enforced Pydantic model at every timestep. The `Action` space validates the agent's findings natively:

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Action(BaseModel):
    missing_values: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of dicts with 'row_idx' and 'col_name'"
    )
    type_errors: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of dicts with 'row_idx' and 'col_name'"
    )
    duplicates: List[int] = Field(
        default_factory=list, 
        description="List of row_idx that are duplicates of previous rows"
    )
    fix_suggestions: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of dicts with 'row_idx', 'col_name', 'suggested_value'"
    )
    done: bool = Field(
        False, 
        description="Set this to true when the audit is complete and want to submit for grading."
    )
```

---

## 3. Observation Space Definition

When the agent triggers an audit cycle or commits to an environment step, it is dynamically supplied with the `Observation` payload representing the present environment conditions:

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class Observation(BaseModel):
    dataset: List[Dict[str, Any]] = Field(
        ..., 
        description="The dataset as a list of dictionaries."
    )
    dataset_schema: Dict[str, str] = Field(
        ..., 
        description="Expected data types for each column."
    )
    task_level: str = Field(
        ..., 
        description="'easy', 'medium', or 'hard'"
    )
    instruction: str = Field(
        ..., 
        description="Instructions for the agent."
    )
```

---

## 4. Task Descriptions

The environment seamlessly modulates difficulty criteria to incrementally test analytical reasoning.

*   **Easy**: Identify all missing values incrementally. The grader performs precision/recall matches on exactly identified parameters via the `missing_values` declaration array.
*   **Medium**: Find structural type errors (e.g., character strings infiltrating numeric integer columns) and identify cascading duplicate rows. Grading emphasizes intersecting the `type_errors` and `duplicates` fields.
*   **Hard**: Full operational audit. Process everything—find the missing values, parse the structural collisions, clear formatting duplicates, and formally render exact remediation resolutions via the `fix_suggestions` schema. Validated via complete strict F1 mapping mathematically bounded between `0.0` - `1.0`.

---

## 5. Setup and Usage Instructions

This environment natively complies with the fast inference `openenv-core` APIs, enabling immediate containerized execution.

**Installation Pipeline**
Ensure Docker or Python >=3.10 is currently installed.
```bash
git clone https://huggingface.co/spaces/MrenikaReddy06/dataset-audit-env
cd dataset-audit-env
pip install -e .
```

**Starting the API Shell**
The built-in FastAPI environment will proxy directly to HuggingFace or locally over Port 8000.
```bash
# Docker Subsystem Execution
docker build -t dataset-audit-env .
docker run -p 8000:8000 dataset-audit-env

# Native Uvicorn Server 
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Executing Local RL Inference**
Evaluate models asynchronously bounding over the target endpoint APIs natively with standard environment variables (`HF_TOKEN`, `API_BASE_URL`):
```bash
python inference.py
```

---

## 6. Baseline Scores

This task calculates deterministic validation scores accurately assessing the agent's inference reasoning directly mirroring human accuracy. Baseline scores under the `hard` paradigm are as follows:

*   **Model Tested**: `Qwen/Qwen2.5-72B-Instruct`
*   **Overall F1 Capability Score**: `0.6667`
