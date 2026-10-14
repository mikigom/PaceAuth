"""Pure decision logic of the API (docs/IMPLEMENTATION.md §6). No I/O; tests in tests/test_api_policy.py.

Every check is written as `not (value >= threshold)` so NaN or garbage always rejects (fail closed).
"""
import math

import numpy as np

DIRECTIONS = ("C", "L", "R", "U", "D", "UL", "UR", "DL", "DR")  # C + 8 peripheral bins (faceauth/terminal/guide.py)
GROUPS = (("C",), ("L", "UL", "DL"), ("R", "UR", "DR"), ("U", "D"))  # template slots 0..3 (subject's left/right)

# internal reason (sessions.reason / audit detail) -> the only reasons a device ever sees
_EXTERNAL = {"ok": "ok", "no_match": "not_recognized", "ambiguous": "not_recognized",
             "subject_inactive": "not_recognized", "challenge_mismatch": "session", "coverage": "coverage",
             "same_person": "retry", "consent": "consent", "already_enrolled": "already_enrolled", "review": "review"}


def external(reason: str) -> str:
    if reason in _EXTERNAL:
        return _EXTERNAL[reason]
    if reason.startswith("session"):
        return "session"
    if reason.startswith("quality:"):
        return "retry"
    return "denied"  # integrity, PAD, assertion, anything unexpected


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not n > 1e-6:  # also rejects NaN
        raise ValueError("degenerate_vector")
    return (v / n).astype(np.float32)


def embeddings(frames: list, dim: int) -> np.ndarray:
    """Engine frames -> (n, dim) float32 L2-normalised. ValueError if any embedding is missing or malformed."""
    out = []
    for f in frames:
        b = f.get("emb")
        if not isinstance(b, bytes) or len(b) != 4 * dim or dim <= 0:
            raise ValueError("bad_embedding")
        out.append(unit(np.frombuffer(b, "<f4")))
    if not out:
        raise ValueError("no_embeddings")
    return np.stack(out)


def frame_reasons(frames: list) -> list:
    """Union of failure reasons; [] iff every frame is ok with exactly one face and an embedding (FR-03)."""
    bad = set()
    for f in frames:
        if not (f.get("ok") is True and f.get("n_faces") == 1 and f.get("emb")):
            bad.update(str(r) for r in (f.get("reasons") or ["not_ok"]))
    return sorted(bad) if frames else ["no_frames"]


def pad_live(frames: list) -> bool:
    """Every frame live. A PAD failure is final, whatever the similarity."""
    return bool(frames) and all(isinstance(f.get("pad"), dict) and f["pad"].get("live") is True for f in frames)


def identify(results: list, t_id: float, margin: float):
    """Search results [{"subject_id", "score"}] (finite scores) -> (subject_id | None, reason, top1, top2).
    Approve only when top1 >= t_id AND top1 - top2 >= margin: top-1 alone never suffices."""
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    top1 = ranked[0]["score"] if ranked else -1.0
    top2 = ranked[1]["score"] if len(ranked) > 1 else -1.0
    if not top1 >= t_id:
        return None, "no_match", top1, top2
    if not top1 - top2 >= margin:
        return None, "ambiguous", top1, top2
    return ranked[0]["subject_id"], "ok", top1, top2


def coverage(poses: list, direction_of) -> tuple:
    """Server-side pose coverage. poses: [yaw, pitch, roll] | None per frame -> (direction per frame, {dir: bool})."""
    dirs = []
    for p in poses:
        try:
            yaw, pitch = float(p[0]), float(p[1])
            dirs.append(direction_of(yaw, pitch) if math.isfinite(yaw) and math.isfinite(pitch) else None)
        except (TypeError, ValueError, IndexError):
            dirs.append(None)
    return dirs, {d: d in dirs for d in DIRECTIONS}


def same_person(embs: np.ndarray, t_same: float) -> tuple:
    """Every embedding has cos >= t_same to the normalised mean -> (ok, min cos)."""
    cos = embs @ unit(embs.mean(0))
    low = float(cos.min())
    return bool(low >= t_same), low


def same_person_sequence(embs: np.ndarray, t_same: float) -> tuple:
    """All selected authentication frames must match each other, even when the largest face changes.

    Similarity to their mean is insufficient: two unrelated identities can each have 0.707 similarity
    to their average. Pairwise comparison prevents cross-person averaging before identity search.
    """
    low = float((embs @ embs.T).min())
    return bool(low >= t_same), low


def templates(embs: np.ndarray, dirs: list) -> list:
    """4 templates: normalised mean of each slot group's frames; a missing group gets the overall mean."""
    overall = unit(embs.mean(0))
    out = []
    for group in GROUPS:
        sel = [e for e, d in zip(embs, dirs) if d in group]
        out.append(unit(np.mean(sel, 0)) if sel else overall)
    return out
