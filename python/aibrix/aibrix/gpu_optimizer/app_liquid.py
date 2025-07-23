import os
import logging
import redis
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
import uvicorn
import argparse
from typing import Dict, Optional, Tuple

from aibrix.gpu_optimizer.load_monitor.load_reader import GatewayLoadReader
from aibrix.gpu_optimizer.load_monitor.profile_reader import RedisProfileReader
from aibrix.gpu_optimizer.load_monitor.monitor_liquid import ModelMonitor

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=0, password=REDIS_PASSWORD
)

logger = logging.getLogger("metrics_server")
logging.basicConfig(level=logging.INFO)

model_monitors: Dict[str, ModelMonitor] = {}
model_names = os.getenv("MODELS", "").split(",")

for model_name in model_names:
    model_name = model_name.strip()
    if not model_name:
        continue
    try:
        reader = GatewayLoadReader(redis_client, model_name)
        profile = RedisProfileReader(redis_client, model_name)
        monitor = ModelMonitor(
            model_name,
            load_reader=reader,
            profile_reader=profile,
            debug=False,
            gpu_fraction=1.0
        )
        monitor.start()
        model_monitors[model_name] = monitor
        logger.info(f"Model monitor started for {model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize monitor for {model_name}: {e}")


async def get_optimizer_metrics(request):
    model_name = request.query_params["model_name"]
    monitor = model_monitors.get(model_name)
    if monitor is None:
        return JSONResponse({"error": f"Model {model_name} not monitored"}, status_code=404)

    try:
        replicas = monitor.read_num_replicas()
        return JSONResponse({
            "model_name": model_name,
            "suggested_replicas": replicas
        })
# Prometheus metrics format
#         metrics_output = f"""# HELP vllm:deployment_replicas Number of suggested replicas.
# # TYPE vllm:deployment_replicas gauge
# vllm:deployment_replicas{{model_name=\"{model_name}\"}} {replicas}
# """
#         return PlainTextResponse(metrics_output)
    except Exception as e:
        logger.error(f"Failed to read replicas for {model_name}: {e}")
        return JSONResponse({"error": f"Failed to read metrics: {e}"}, status_code=500)


app = Starlette(
    routes=[
        Route("/optimizer", get_optimizer_metrics),
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the optimizer metrics server")
    parser.add_argument("--port", type=int, default=8090, help="Port to run the server on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)