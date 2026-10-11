"""E2: 1:N open-set identification harness (protocol builder, exact GPU search, metrics + CIs, extrapolation).

    python -m eval.onen.protocol ...   # build reports/E2_1N/protocol/<name>/
    python -m eval.onen.run ...        # embed (or load .npy) + search + metrics -> reports/E2_1N/<model>/
    python -m eval.onen.synth ...      # harness validation on synthetic identities with known ground truth
"""
