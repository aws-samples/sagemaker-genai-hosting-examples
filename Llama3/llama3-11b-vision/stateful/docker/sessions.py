import argparse
import logging
import sys
import uuid
import torch
import boto3
from pathlib import Path

from torch.distributed import FileStore


OPEN_SESSION = "open_sessions"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--store_folder",
        dest="store_folder",
        help="Folder for store",
        type=str,
        default="/tmp",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="Model name",
        type=str,
        default="model_name",
    )

    args = parser.parse_args()

    cloudwatch = boto3.client("cloudwatch", region_name="us-west-2")

    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    gpu_memory_metrics = []
    for gpu_index in range(torch.cuda.device_count()):
        mem_get_info = torch.cuda.mem_get_info(gpu_index)
        mem_used_mib = (mem_get_info[1] - mem_get_info[0])/(1024*1024)
        gpu_memory_metrics.append(
            {
                "MetricName": "GPUMemoryUtilizationInMiB",
                "Dimensions": [
                    {
                        "Name": "GPUId",
                        "Value": str(gpu_index)
                    },
                    {
                        "Name": "InstanceId",
                        "Value": str(uuid.getnode())
                    }
                ],
                "Unit": "Megabytes",
                "StorageResolution": 60,
                "Value": mem_used_mib
            }
        )
        logging.info(f"GPUMemoryUtilizationInMiB: {mem_used_mib} GPUId: {gpu_index} InstanceId: {uuid.getnode()}")

    cloudwatch.put_metric_data(MetricData=gpu_memory_metrics, Namespace="metaDemo")

    store_path = Path(args.store_folder) / f"{args.model_name}_store"
    if not store_path.exists():
        logging.info("FileStore does not exist yet: " + str(store_path))
        return

    store = FileStore(store_path.as_posix(), -1)

    open_sessions = store.compare_set(OPEN_SESSION, "EMPTY", "").decode("utf-8").strip().strip(";")
    open_sessions_count = 0
    if open_sessions != "" and open_sessions != "EMPTY":
        open_sessions_count = len(open_sessions.split(";"))

    cloudwatch.put_metric_data(
        MetricData=[
            {
                "MetricName": "OpenSessionsCount",
                "Dimensions": [
                    {
                        "Name": "InstanceId",
                        "Value": str(uuid.getnode())
                    }
                ],
                "Unit": "Count",
                "StorageResolution": 60,
                "Value": open_sessions_count
            }
        ],
        Namespace="metaDemo"
    )

    logging.info(f"OpenSessionsCount: {open_sessions_count} InstanceId: {uuid.getnode()}")


if __name__ == "__main__":
    main()
