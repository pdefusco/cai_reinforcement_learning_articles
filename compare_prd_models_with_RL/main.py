


import uuid
import math
import asyncio
from fastapi import FastAPI, HTTPException
import httpx
from db import init_db, get_conn
from models import InferenceRequest, InferenceResponse

app = FastAPI(title="RL Model Router - UCB (Hardcoded Models)")

# --------------------------------------------------
# Hardcoded Model Registry
# --------------------------------------------------

MODEL_REGISTRY = {
    "xgb_depth4": {
        "endpoint": "https://model-a.cloudera.ai/predict",
        "token": "TOKEN_MODEL_A",
        "pulls": 0,
        "total_reward": 0.0
    },
    "xgb_depth8": {
        "endpoint": "https://model-b.cloudera.ai/predict",
        "token": "TOKEN_MODEL_B",
        "pulls": 0,
        "total_reward": 0.0
    },
    "xgb_lr01": {
        "endpoint": "https://model-c.cloudera.ai/predict",
        "token": "TOKEN_MODEL_C",
        "pulls": 0,
        "total_reward": 0.0
    }
}

# --------------------------------------------------
# Multi-Armed Bandit (UCB1)
# --------------------------------------------------

def select_model_ucb():

    total_pulls = sum([m["pulls"] for m in MODEL_REGISTRY.values()])

    # Force initial exploration
    for name, model in MODEL_REGISTRY.items():
        if model["pulls"] == 0:
            return name, model

    best_score = float("-inf")
    best_model_name = None
    best_model = None

    for name, model in MODEL_REGISTRY.items():
        avg_reward = model["total_reward"] / model["pulls"]
        confidence = math.sqrt((2 * math.log(total_pulls)) / model["pulls"])
        ucb_score = avg_reward + confidence

        if ucb_score > best_score:
            best_score = ucb_score
            best_model_name = name
            best_model = model

    return best_model_name, best_model


# --------------------------------------------------
# Call Cloudera Inference Service
# --------------------------------------------------

async def call_model(model_config, payload):

    headers = {
        "Authorization": f"Bearer {model_config['token']}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            model_config["endpoint"],
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()["prediction"]


# --------------------------------------------------
# FastAPI Startup
# --------------------------------------------------

@app.on_event("startup")
async def startup():
    init_db()
    asyncio.create_task(reward_updater())


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):

    model_name, model_config = select_model_ucb()

    try:
        prediction = await call_model(model_config, request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    request_id = str(uuid.uuid4())

    # Store prediction only (bandit stats are in memory)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO predictions (request_id, model_name, prediction)
            VALUES (?, ?, ?)
        """, (request_id, model_name, prediction))

    return InferenceResponse(
        request_id=request_id,
        model_used=model_name,
        prediction=prediction
    )


# --------------------------------------------------
# Metrics Endpoint (Demo Visualization)
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    total_pulls = sum([m["pulls"] for m in MODEL_REGISTRY.values()]) or 1

    results = []

    for name, model in MODEL_REGISTRY.items():

        pulls = model["pulls"]
        avg_reward = model["total_reward"] / pulls if pulls > 0 else 0
        exploration_bonus = (
            math.sqrt((2 * math.log(total_pulls)) / pulls)
            if pulls > 0 else "∞"
        )

        results.append({
            "model": name,
            "pulls": pulls,
            "avg_reward": avg_reward,
            "ucb_bonus": exploration_bonus
        })

    return results


# --------------------------------------------------
# Reward Updater
# --------------------------------------------------

async def reward_updater():

    while True:
        await asyncio.sleep(5)

        with get_conn() as conn:
            c = conn.cursor()

            c.execute("""
                SELECT id, model_name, prediction, ground_truth
                FROM predictions
                WHERE reward IS NULL AND ground_truth IS NOT NULL
            """)

            rows = c.fetchall()

            for pred_id, model_name, pred, truth in rows:

                reward = -abs(pred - truth)

                # Update prediction table
                c.execute("""
                    UPDATE predictions
                    SET reward = ?
                    WHERE id = ?
                """, (reward, pred_id))

                # Update in-memory bandit stats
                MODEL_REGISTRY[model_name]["pulls"] += 1
                MODEL_REGISTRY[model_name]["total_reward"] += reward
