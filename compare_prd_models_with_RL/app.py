#****************************************************************************
# Streamlit RL Routing Simulator
#****************************************************************************

import streamlit as st
import math
import random
import time
import httpx
from pyspark.sql import SparkSession

# --------------------------------------------------
# Spark Session (Placeholder)
# --------------------------------------------------

spark = SparkSession.builder.appName("RL-Routing-Demo").getOrCreate()

# Placeholder Spark SQL table
# Replace with your real training dataset table
DATA_TABLE = "training_data_placeholder"

# Example schema assumption:
# features: array<double>
# label: double

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
    }
}

# --------------------------------------------------
# Multi-Armed Bandit (UCB1)
# --------------------------------------------------

def select_model_ucb():
    total_pulls = sum([m["pulls"] for m in MODEL_REGISTRY.values()])

    # Force exploration initially
    for name, model in MODEL_REGISTRY.items():
        if model["pulls"] == 0:
            return name, model

    best_score = float("-inf")
    best_name = None
    best_model = None

    for name, model in MODEL_REGISTRY.items():
        avg_reward = model["total_reward"] / model["pulls"]
        confidence = math.sqrt((2 * math.log(total_pulls)) / model["pulls"])
        score = avg_reward + confidence

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    return best_name, best_model


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
# Simulate Requests
# --------------------------------------------------

async def simulate_requests(num_requests):

    # Sample data from Spark SQL
    df = spark.sql(f"SELECT * FROM {DATA_TABLE} LIMIT {num_requests}")
    rows = df.collect()

    traffic_placeholder = st.empty()
    reward_placeholder = st.empty()

    for i, row in enumerate(rows):

        features = row["features"]
        ground_truth = row["label"]

        model_name, model_config = select_model_ucb()

        prediction = await call_model(
            model_config,
            {"features": features}
        )

        reward = -abs(prediction - ground_truth)

        # Update bandit stats
        model_config["pulls"] += 1
        model_config["total_reward"] += reward

        # Update UI every 50 requests for smoother rendering
        if i % 50 == 0:

            traffic_data = {
                name: m["pulls"]
                for name, m in MODEL_REGISTRY.items()
            }

            reward_data = {
                name: (
                    m["total_reward"] / m["pulls"]
                    if m["pulls"] > 0 else 0
                )
                for name, m in MODEL_REGISTRY.items()
            }

            traffic_placeholder.bar_chart(traffic_data)
            reward_placeholder.bar_chart(reward_data)

            time.sleep(0.05)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("Multi-Armed Bandit Routing Demo (UCB)")
st.write("Simulating traffic between two XGBoost models deployed in Cloudera Inference Service.")

num_requests = st.slider(
    "Number of simulated requests",
    min_value=100,
    max_value=5000,
    value=2000,
    step=100
)

if st.button("Run Simulation"):

    # Reset bandit state
    for model in MODEL_REGISTRY.values():
        model["pulls"] = 0
        model["total_reward"] = 0.0

    st.write("Running simulation...")
    import asyncio
    asyncio.run(simulate_requests(num_requests))

    st.success("Simulation complete.")
