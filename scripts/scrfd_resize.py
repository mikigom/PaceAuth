"""Re-target the static SCRFD-34G export (1x3x640x640) to another square input size, same weights.

    python scripts/scrfd_resize.py --size 320 --out data/models/scrfd_34g_320

Why (E9): server-side re-detection at 640 used ~70% of the GPU per frame; kiosk faces at 40-80 cm are 120-250 px in
a 640x480 frame, so a 320 input still sees them (and background faces of >= ~25 px). Only size-dependent constants
change: the two FPN Resize target sizes and the GroupNorm Reshape back-shapes (feature maps = size / 8, 16, 32),
plus the declared input/output shapes. The result is a new registry version of `scrfd_34g` (sign + activate it).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

SRC = Path("data/models/scrfd_34g")


def resize(model: onnx.ModelProto, size: int) -> onnx.ModelProto:
    if size % 32:
        raise SystemExit("size must be a multiple of 32")
    old = {640 // s: size // s for s in (8, 16, 32)}            # 80 -> size/8, 40 -> size/16, 20 -> size/32
    g = model.graph
    targets = {n.input[3] for n in g.node if n.op_type == "Resize" and len(n.input) > 3}
    targets |= {n.input[1] for n in g.node if n.op_type == "Reshape"}
    changed = 0
    for t in list(g.initializer):
        if t.name in targets:
            a = numpy_helper.to_array(t)
            if a.ndim == 1 and a.size == 4 and a[2] == a[3] and int(a[2]) in old:
                t.CopyFrom(numpy_helper.from_array(np.array([*a[:2], old[int(a[2])], old[int(a[3])]], a.dtype), t.name))
                changed += 1
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] in targets:
            a = numpy_helper.to_array(n.attribute[0].t)
            if a.ndim == 1 and a.size == 4 and a[2] == a[3] and int(a[2]) in old:
                n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array([*a[:2], old[int(a[2])], old[int(a[3])]], a.dtype)))
                changed += 1
    if changed != 5:   # 2 Resize sizes + 3 GroupNorm back-shapes (the two GN layers of a stride share one constant)
        raise SystemExit(f"expected 5 unique size constants, changed {changed}")
    dims = g.input[0].type.tensor_type.shape.dim
    dims[2].dim_value = dims[3].dim_value = size
    for o in g.output:                                          # (anchors, c): anchors = 2 * (size/stride)^2
        stride = int(o.name.split("_")[1])
        o.type.tensor_type.shape.dim[0].dim_value = 2 * (size // stride) ** 2
    del g.value_info[:]                                         # stale 640 intermediate shapes; re-infer below
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("data/models/scrfd_34g_320"))
    a = ap.parse_args()
    meta = json.loads((SRC / "MODEL.json").read_text())
    src_sha = dict(meta["sha256"])
    m = resize(onnx.load(SRC / meta["files"]["onnx"]), a.size)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"scrfd_34g_gnkps_{a.size}.onnx"
    onnx.save(m, a.out / name)
    meta.update(version="2", files={"onnx": name},
                sha256={"onnx": hashlib.sha256((a.out / name).read_bytes()).hexdigest()})
    meta["params"] = dict(meta.get("params", {}), input_size=a.size, input=f"1x3x{a.size}x{a.size}",
                          derived_from=f"scrfd_34g@1 onnx sha256 {src_sha['onnx']}", derivation="scripts/scrfd_resize.py")
    (a.out / "MODEL.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {a.out / name} (version 2, input {a.size})")


if __name__ == "__main__":
    main()
