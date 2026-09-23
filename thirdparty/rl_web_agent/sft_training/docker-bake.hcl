// You must build or provide the base image yourself and point REGISTRY at your own registry.
variable "REGISTRY" {
  default = "your-registry.example.com/rl-web-agent-base"
}

group "default" {
  targets = ["nemo"]
}

target "nemo" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    REPO = REGISTRY
    BASE_TAG = "base"
  }
  tags = ["${REGISTRY}:nemo"]
  platforms = ["linux/amd64"]

  # Cache sources (import)
  cache-from = [
    "type=local,src=./.buildx-cache"
  ]

  # Cache destination (export)
  cache-to = [
    "type=local,dest=./.buildx-cache,mode=max"
  ]
  # cache-to = ["type=registry,ref=${REGISTRY}:cache,mode=max"]
  # cache-from = ["type=registry,ref=${REGISTRY}:cache"]
  push = true
}
