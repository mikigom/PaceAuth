#!/usr/bin/env bash
# Component list (RFP §6 SBOM): base images by digest, Python packages in faceauth:dev, signed model registry entries,
# datasets used for evaluation. -> reports/SBOM.json. Run on the Docker host from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for img in faceauth:dev pytorch/pytorch:2.12.0-cuda13.0-cudnn9-devel postgres:16-alpine nginx:alpine; do
  docker image inspect "$img" --format '{"image":"{{index .RepoTags 0}}","id":"{{.Id}}","digest":"{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}","created":"{{.Created}}"}'
done > "$tmp/images.jsonl"
docker run --rm faceauth:dev pip list --format=json > "$tmp/pip.json"
cp "$tmp"/images.jsonl "$tmp"/pip.json data/ 2>/dev/null || true
docker run --rm -i -u "$(id -u):$(id -g)" -v "$PWD:/app" -w /app faceauth:dev python - <<'EOF'
import json, pathlib
images = [json.loads(l) for l in open("data/images.jsonl")]
pip = json.load(open("data/pip.json"))
reg = json.loads(pathlib.Path("data/models/registry.json").read_text())["models"]
models = [{k: m.get(k) for k in ("name", "version", "kind", "sha256", "source", "license")} for m in reg]
datasets = []
for p in sorted(pathlib.Path("data/datasets").glob("*/DATASET.json")):
    d = json.loads(p.read_text())
    datasets.append({k: d.get(k) for k in ("name", "version", "source", "license")})
out = {"images": images, "python_packages": pip, "models": models, "datasets": datasets,
       "note": "Model weights and datasets are research/non-commercial unless stated; see each licence field."}
pathlib.Path("reports/SBOM.json").write_text(json.dumps(out, indent=1))
print(f"reports/SBOM.json: {len(images)} images, {len(pip)} packages, {len(models)} models, {len(datasets)} datasets")
EOF
rm -f data/images.jsonl data/pip.json
