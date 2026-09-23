// Base image: the slime/sglang CUDA base image (see docs/environment-setup.md).
// You must build or provide this image yourself and point REGISTRY at your own registry.
variable "REGISTRY" {
  default = "your-registry.example.com/slime-base"
}

group "default" {
  targets = ["slime_rl"]
}

target "slime_rl" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    REPO = REGISTRY
    BASE_TAG = "base"
  }
  tags = ["${REGISTRY}:slime_rl"]
  platforms = ["linux/amd64"]
  push = true
}
