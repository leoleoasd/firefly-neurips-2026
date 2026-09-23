
python -c "
import ray
ray.init(address='auto', namespace='sglang')
registry = ray.get_actor('sglang_registry')
print(ray.get(registry.dump.remote()))
"

python -c "
import ray
ray.init(address='auto', namespace='sglang')
registry = ray.get_actor('sglang_registry')
print('Before:', ray.get(registry.dump.remote()))
ray.get(registry.clear.remote())
print('After:', ray.get(registry.dump.remote()))
"


python3 tool_call_agent/evaluate.py --checkpoint /path/to/your/checkpoint-hf/ --num-gpus 4 --num-tasks 100 --batch 3 --task-file test_tasks.json --tool-mode dag --seed 42 --no-pin-node --passes 8 --long-output-threshold 200000000 --only-missing
