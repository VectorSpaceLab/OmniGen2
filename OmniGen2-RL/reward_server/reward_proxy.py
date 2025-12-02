#!/usr/bin/env python3

from typing import List, Dict, Any, Tuple
import argparse
import pickle
import requests
import json
import time
import logging
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import math
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RewardProxy")

app = Flask(__name__)


def reorder_results(
    merged_item_list: List[Dict[str, Any]], original_batch_size: int, app: Flask
) -> Dict[str, Any]:
    """
    Reorder the merged result list according to the original indices.

    Args:
        merged_item_list: A flattened list, each element is a dict containing a single image result.
                          e.g. [{'score': 0.8, 'meta_data': {'original_index': 5}}, ...]
        original_batch_size: The original batch size of the request.

    Returns:
        A dict containing sorted 'scores', 'rewards', 'meta_data', etc.
    """
    if not merged_item_list:
        logger.warning("Merged result list is empty, cannot reorder results.")
        # Return an empty result in the expected format
        return {
            "scores": [0.0] * original_batch_size,
            "rewards": [0.0] * original_batch_size,
            "reasoning": [""] * original_batch_size,
            "strict_rewards": [0.0] * original_batch_size,
            "meta_data": [
                {"original_index": i, "error": "No result received"}
                for i in range(original_batch_size)
            ],
            "group_rewards": {},
            "group_strict_rewards": {},
        }

    # 1. Create placeholder lists, pre-allocated to the correct size
    ordered_scores = [0.0] * original_batch_size
    ordered_rewards = [0.0] * original_batch_size
    ordered_reasoning = [""] * original_batch_size
    ordered_strict_rewards = [0.0] * original_batch_size
    ordered_meta_datas = [
        {"original_index": i, "error": "Result missing from server response"}
        for i in range(original_batch_size)
    ]

    # 2. Iterate over the flattened result list and place each result in the correct position
    found_count = 0
    for item in merged_item_list:
        if not isinstance(item, dict):
            logger.warning(f"Found non-dict result item, skipped: {item}")
            continue

        meta = item.get("meta_data", {})
        original_index = meta.get("original_index")

        if original_index is not None and 0 <= original_index < original_batch_size:
            ordered_scores[original_index] = item.get("score", 0.0)
            ordered_rewards[original_index] = item.get("reward", 0.0)
            ordered_reasoning[original_index] = item.get("reasoning", "")
            ordered_strict_rewards[original_index] = item.get("strict_reward", 0.0)
            ordered_meta_datas[original_index] = meta
            found_count += 1
        else:
            logger.warning(f"Found invalid or missing 'original_index' in result item: {item}")

    # 4. Logging
    if found_count < original_batch_size:
        logger.warning(
            f"Result reordering incomplete: {found_count}/{original_batch_size} results found."
        )
    else:
        logger.info(f"Result reordering complete: {found_count}/{original_batch_size} matched successfully.")

    return {
        "scores": ordered_scores,
        "rewards": ordered_rewards,
        "reasoning": ordered_reasoning,
        "strict_rewards": ordered_strict_rewards,
        "meta_data": ordered_meta_datas,
    }


class RewardProxy:
    def __init__(self, worker_configs: List[Dict[str, Any]]):
        self.server_urls = self._build_server_urls(worker_configs)
        print(f"{len(self.server_urls)=}, {worker_configs=}", flush=True)
        self.executor = ThreadPoolExecutor(max_workers=len(self.server_urls))

        logger.info("🚀 Proxy initialized")
        logger.info(f"  -> servers {self.server_urls=} ...")

    @staticmethod
    def _build_server_urls(worker_configs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        server_urls = []
        for conf in worker_configs:
            server_urls.extend([f"http://{conf['host']}:{conf['base_port'] + i}" for i in range(conf['num_servers'])])

        return server_urls

    def _send_request_to_worker(
        self, server_url: str, batch_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send request to a single worker server and return the result."""
        try:
            response = requests.post(
                server_url,
                data=pickle.dumps(batch_data),
                headers={"Content-Type": "application/octet-stream"},
                timeout=600,  # 300 seconds timeout
            )
            response.raise_for_status()  # Raise exception for 4xx or 5xx status codes
            return pickle.loads(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to server {server_url} failed: {e}")
        except pickle.PickleError as e:
            logger.error(f"Failed to parse response from {server_url}: {e}")
        return None  # Return None to indicate failure
    
    def rebatch_with_instruction(
        self,
        input_images,
        output_image: List,
        meta_datas: List,
    ):
        input_images_group = defaultdict(list)
        output_image_group = defaultdict(list)
        meta_datas_group = defaultdict(list)

        original_index_group = defaultdict(list)

        for i in range(len(input_images)):
            key = meta_datas[i]["instruction"]
            input_images_group[key].append(input_images[i])
            output_image_group[key].append(output_image[i])
            meta_datas_group[key].append(meta_datas[i])

            original_index_group[key].append(i)

        return input_images_group, output_image_group, meta_datas_group, original_index_group


    def process_batch(
        self,
        input_images,
        output_image: List,
        meta_datas: List,
        **kwargs,
    ) -> Dict[str, List[Any]]:
        """
        Dispatch batch tasks to the specified type of worker servers and merge results.
        """

        input_images_group, output_image_group, meta_datas_group, original_index_group = self.rebatch_with_instruction(input_images, output_image, meta_datas)
        
        num_workers = len(self.server_urls)

        original_index = []
        futures = []
        for i, key in enumerate(input_images_group.keys()):
            server_url = self.server_urls[i % num_workers]
            payload = {
                "input_images": input_images_group[key],
                "output_image": output_image_group[key],
                "meta_data": meta_datas_group[key],
                **kwargs,  # Pass use_flowgrpo, debug, etc.
            }
            # print(f"{server_url=}, {start_idx + i * size_per_worker}:{min(start_idx + (i + 1) * size_per_worker, end_idx)}: {len(payload['input_images'])=}, {len(payload['output_image'])=}, {len(payload['meta_data'])=}", flush=True)
            futures.append(
                self.executor.submit(self._send_request_to_worker, server_url, payload)
            )

            original_index.extend(original_index_group[key])
        
        inverse_original_index = {i: idx for idx, i in enumerate(original_index)}

        # Merge all successful results
        merged_results = []
        for future in futures:
            result = future.result()
            merged_results.extend(result)

        # reorder results by original index
        merged_results = [merged_results[inverse_original_index[i]] for i in range(len(original_index))]

        return merged_results


def prepare_request_data(request_body: bytes) -> Tuple[List, List, str, Dict]:
    """Parse request body and add original index to meta data."""
    data = pickle.loads(request_body)
    input_images = data["input_images"]
    output_image = data["output_image"]
    meta_datas = data["meta_datas"]

    meta_datas = [json.loads(meta) for meta in meta_datas]

    # Add original index to each meta_data for later sorting
    for i, meta in enumerate(meta_datas):
        meta["original_index"] = i

    server_type = data.get("server_type", "geneval")
    return input_images, output_image, meta_datas, server_type


# Flask route
@app.route("/", methods=["POST"])
def evaluate():
    try:
        input_images, output_image, meta_datas, server_type = prepare_request_data(
            request.data
        )
        original_batch_size = len(output_image)
        logger.info(
            f"Received evaluation request: {original_batch_size} images, server type: {server_type}"
        )
    except Exception as e:
        logger.error(f"Failed to parse request: {e}", exc_info=True)
        # Return a JSON error, more universal than pickle
        return jsonify(
            {"error": "Failed to parse request data", "details": str(e)}
        ), 400

    start_time = time.time()

    proxy = app.proxy
    # Dispatch processing
    merged_results = proxy.process_batch(
        input_images, output_image, meta_datas
    )

    # Reorder results by index
    ordered_result = reorder_results(merged_results, original_batch_size, app)

    total_time = time.time() - start_time
    logger.info(
        f"Evaluation complete! Total time: {total_time:.3f}s ({total_time / original_batch_size * 1000:.1f} ms/image)"
    )

    return pickle.dumps(ordered_result)


def main():
    parser = argparse.ArgumentParser(description="Universal Reward Proxy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument(
        "--config_path",
        type=str,
        default="server_configs/editscore_7B.yml",
        help="Configuration file path",
    )
    # parser.add_argument("--port", type=int, default=23456, help="Proxy server port")

    # parser.add_argument("--worker_host", type=str, default="127.0.0.1")
    # parser.add_argument("--worker_base_port", type=int, default=18888)
    # parser.add_argument("--worker_num_machines", type=int, default=1)
    # parser.add_argument("--max_workers_per_machine", type=int, default=128)
    # parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # print(
    #     f"{args.worker_host=}, {args.worker_base_port=}, {args.worker_num_machines=}, {args.max_workers_per_machine=}, {args.batch_size=}",
    #     flush=True,
    # )
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    proxy_port = config["server"]["proxy_port"]

    worker_configs = []

    hosts = config["server"]["hosts"]
    for i, host in enumerate(hosts):
        worker_configs.append(
            {
                "host": host,
                "base_port": config["server"]["worker_base_port"],
                "num_servers": 8 // config["reward"]["tensor_parallel_size"],
            }
        )

    proxy_instance = RewardProxy(worker_configs)
    app.proxy = proxy_instance

    logger.info(f"Starting proxy server at {worker_configs=}")

    app.run(
        host=args.host, port=proxy_port, debug=False, threaded=True, use_reloader=False
    )


if __name__ == "__main__":
    main()
