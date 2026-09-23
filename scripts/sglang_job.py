"""
ray job submit --address=auto \
  --working-dir . \
  -- python scripts/sglang_job.py --num-nodes 8 --num-gpus 1 --model /tmp/instance_storage/qwen3-30B-A3B --tp 1 --reasoning-parser deepseek-r1 --tool-call-parser qwen


ray job submit --address=auto \
  --working-dir . \
  -- python scripts/sglang_job.py --num-gpus 8 --num-nodes 3 \
    --model /tmp/instance_storage/Qwen/Qwen3-235B-A22B-Thinking-2507 \
    --context-length 131072  --reasoning-parser deepseek-r1  --tool-call-parser qwen \
    --tp 8


ray job submit --address=auto \
  --working-dir . \
  -- python scripts/sglang_job.py --num-gpus 1 --num-nodes 1 --node-ip $(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}') \
    --model /tmp/instance_storage/after_merge_ckpt/ --tool-call-parser qwen

"""

import argparse
import logging
import multiprocessing
import time

import ray
import requests
from sglang.srt.server_args import ServerArgs
from slime.utils.misc import get_current_node_ip

from shared.sglang_registry import get_or_create_registry
from shared.utils import get_random_free_port

logger = logging.getLogger(__name__)

# Registry keys
WORKER_REGISTRY_KEY = "rm_worker"
ROUTER_REGISTRY_KEY = "rm_router"


def _wait_server_healthy(base_url: str, api_key: str | None, is_process_alive):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with requests.Session() as session:
        while True:
            try:
                r = session.get(f"{base_url}/health_generate", headers=headers, timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass

            if not is_process_alive():
                raise RuntimeError("SGLang server process terminated unexpectedly.")

            time.sleep(2)


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    """
    Start SGLang server using the same pattern as slime:
    multiprocessing spawn -> sglang.srt.entrypoints.http_server.launch_server
    """
    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    # Single-node reward server: node_rank is effectively 0, so we can health-check here.
    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=getattr(server_args, "api_key", None),
        is_process_alive=lambda: p.is_alive(),
    )

    return p


@ray.remote
class RewardSGLangActor:
    def __init__(self, args, registry_name: str):
        self.args = args
        self.registry_name = registry_name

        self.proc: multiprocessing.Process | None = None
        self.url: str | None = None

    def start(self) -> str:
        server_args = ServerArgs.from_cli_args(self.args)

        node_ip = get_current_node_ip()
        port = get_random_free_port()

        # Force host/port policy
        server_args.host = "0.0.0.0"
        server_args.port = port

        self.proc = launch_server_process(server_args)
        self.url = f"http://{node_ip}:{port}"
        return self.url

    def register(self) -> None:
        """Register as rm_worker and add self to rm_router."""
        if self.url is None:
            raise RuntimeError("start() must be called before register().")

        reg = get_or_create_registry(self.registry_name)

        # 1. Register self as rm_worker
        ray.get(reg.add.remote(WORKER_REGISTRY_KEY, self.url))
        logger.info(f"Registered as {WORKER_REGISTRY_KEY}: {self.url}")

        # 2. Get router URL and add self to router
        router_urls = ray.get(reg.get_all.remote(ROUTER_REGISTRY_KEY))
        if router_urls:
            router_url = router_urls[0]
            self._add_to_router(router_url)
        else:
            logger.warning(f"No {ROUTER_REGISTRY_KEY} found in registry, skipping router registration")

    def _add_to_router(self, router_url: str) -> None:
        """Add this worker to the sglang router via POST /workers."""
        try:
            resp = requests.post(
                f"{router_url}/workers",
                json={"url": self.url},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info(f"Added to router {router_url}: {self.url}")
        except Exception as e:
            logger.error(f"Failed to add to router {router_url}: {e}")

    def wait_forever(self) -> None:
        if self.proc is None:
            raise RuntimeError("start() must be called before wait_forever().")

        while True:
            if not self.proc.is_alive():
                raise RuntimeError("Reward SGLang server died.")
            time.sleep(10)


def parse_args():
    parser = argparse.ArgumentParser("Ray Job: start SGLang reward model(s)")

    # Launcher-specific args
    parser.add_argument("--num-gpus", type=float, default=1)
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of independent SGLang instances to launch in parallel (default: 1)",
    )
    parser.add_argument("--registry-name", default="sglang_registry")
    parser.add_argument(
        "--node-ip",
        type=str,
        default=None,
        help="Pin all actors to the node with this IP address",
    )

    # Inject ALL SGLang server args for this installed version
    ServerArgs.add_cli_args(parser)

    args = parser.parse_args()

    # Default tp to num_gpus if user didn't explicitly set --tp / --tensor-parallel-size
    if args.tensor_parallel_size == 1 and args.num_gpus > 1:
        args.tensor_parallel_size = int(args.num_gpus)

    return args


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    ray.init(address="auto")

    num_nodes = args.num_nodes

    # Launch N independent actors in parallel
    actors = []
    actor_options = {
        "num_gpus": args.num_gpus,
        "num_cpus": 1,
    }
    if args.node_ip:
        actor_options["resources"] = {f"node:{args.node_ip}": 0.001}
        print(f"[RewardSGLang] --node-ip: pinning all actors to node {args.node_ip}")

    for _i in range(num_nodes):
        actor = RewardSGLangActor.options(**actor_options).remote(
            args=args,
            registry_name=args.registry_name,
        )
        actors.append(actor)

    # Start all actors in parallel
    start_futures = [actor.start.remote() for actor in actors]
    urls = ray.get(start_futures)
    for i, url in enumerate(urls):
        print(f"[RewardSGLang] node {i}: URL={url}", flush=True)

    # Register all actors in parallel
    register_futures = [actor.register.remote() for actor in actors]
    ray.get(register_futures)
    for i, url in enumerate(urls):
        print(
            f"[RewardSGLang] node {i}: registered as {WORKER_REGISTRY_KEY}: {url}",
            flush=True,
        )

    # Ensure registry health check is running
    registry = get_or_create_registry(args.registry_name)
    ray.get(registry.start_health_check.remote())
    print("[RewardSGLang] registry health check started", flush=True)

    # Wait on all actors (block until any dies)
    wait_futures = [actor.wait_forever.remote() for actor in actors]
    ray.get(wait_futures)


if __name__ == "__main__":
    main()
