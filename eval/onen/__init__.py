"""E2 1:N open-set identification harness and E6 alignment ablation.

    datasets.py  identity / distractor sources (CelebA, FFHQ, DigiFace-1M zips, VGGFace2-test)
    embed.py     resumable sharded detection + alignment (shared crops) and per-model embedding
    protocol.py  seeded subject-disjoint protocol builder
    search.py    exact GPU search (faiss), subject aggregation, score histograms
    metrics.py   FPIR/FNIR/misID with margin rule, Clopper-Pearson + cluster bootstrap, DET, GPD tail extrapolation
    run.py       orchestrator: python -m eval.onen.run {align,protocol,digiface,embed,evaluate,e6-*,summary}
    report.py    Korean summaries (reports/E2_1N/summary.md, reports/E6_alignment/summary.md)
"""
