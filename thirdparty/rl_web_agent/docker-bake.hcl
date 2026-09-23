// You must build or provide the base image yourself and point REGISTRY at your own registry.
variable "REGISTRY" {
  default = "your-registry.example.com/rl-web-agent-base"
}

group "default" {
  targets = ["sglang"]
}

target "sglang" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    ROLLOUT_ENGINE = "sglang"
    REPO = REGISTRY
    BASE_TAG = "base"
  }
  tags = ["${REGISTRY}:verl_sglang"]
  platforms = ["linux/amd64"]
  cache-to = ["type=registry,ref=${REGISTRY}:cache,mode=max"]
  cache-from = ["type=registry,ref=${REGISTRY}:cache"]
  push = true
}
