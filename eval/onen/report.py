"""E2 / E6 Korean summaries from reports/E2_1N/<model>/results.json (+ per-search arrays in data/tmp/recog/onen/emb/).

    python -m eval.onen.run summary

Threshold recommendation for the deployed model (rule fixed in code, applied to whatever the numbers are):
  FA_T_ID   = max(extrapolated threshold for FPIR 1e-6 at 1M x 4 (main GPD fit), measured threshold at the lowest
              resolvable FPIR on the largest gallery), rounded UP to 0.01 (RFP §4: FPIR target first).
  FA_MARGIN = largest grid margin whose FNIR cost on the largest gallery is <= 0.1 pp both at FA_T_ID and at the
              FPIR=1e-3 operating point.
"""
import json
import math

import numpy as np

from eval.onen import metrics
from eval.onen.run import (DEPLOYED, MARGINS, MODELS, REPORT, REPORT_E6, TARGET_G, TARGET_TPL, WORK,
                           operating_point)

TRAIN = {"kprpe_vitb_wf12m": "ViT-B KP-RPE · WF12M", "insightface_r50_wf600k": "R50 · WF600K (buffalo_l)",
         "insightface_r100_glint360k": "R100 · Glint360K (antelopev2)", "adaface_ir101_wf12m": "IR101 AdaFace · WF12M"}
MARGIN_COST = 0.001
N_TPL = TARGET_G * TARGET_TPL


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def ci(c, d=2):
    return f"[{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


def grid(r, gi=-1, v="single", m="0.0", pool=None):
    ev = r["galleries"][gi]["variants"][v]
    rows = ev["other_pools"][pool]["margins"]["0.0"]["grid"] if pool else ev["margins"][m]["grid"]
    return {row["target_fpir"]: row for row in rows}


def searches(model, D, v):
    z = np.load(WORK / "emb" / model / f"searches_{D}_{v}.npz")
    return {k: z[k] for k in z.files}


def op(model, D, v, t, m):
    return operating_point(searches(model, D, v), t, m)


def real_tail(model):
    h = np.load(WORK / "emb" / model / "impostor_hist.npz")
    return metrics.ImpostorTail(h["celeba__celeba"] + h["celeba__ffhq"])


def recommend(model, r):
    ext = r["extrapolation_1M_x4"]
    res_rows = [row for row in grid(r).values() if row["resolvable"]]
    t_meas = res_rows[-1]["threshold"]
    t_id = math.ceil(100 * max(ext["threshold"], t_meas) - 1e-9) / 100
    D = r["galleries"][-1]["distractor_subjects"]
    t3 = grid(r)[1e-3]["threshold"]
    rows = {m: op(model, D, "single", t_id, m) for m in MARGINS}
    rows3 = {m: op(model, D, "single", t3, m) for m in MARGINS}
    ok = [m for m in MARGINS if rows[m]["fnir"] - rows[0.0]["fnir"] <= MARGIN_COST
          and rows3[m]["fnir"] - rows3[0.0]["fnir"] <= MARGIN_COST]
    sens = [x["threshold"] for x in ext["gpd_threshold_sensitivity"]]
    return {"t_ext": ext["threshold"], "t_ext_range": (min(sens), max(sens)), "t_meas": t_meas,
            "fpir_meas": res_rows[-1]["target_fpir"], "t_id": t_id, "margin": max(ok), "rows": rows, "D": D}


def e2_summary():
    res = {m: json.loads((REPORT / m / "results.json").read_text()) for m in MODELS if (REPORT / m / "results.json").exists()}
    proto = json.loads((REPORT / "protocol.json").read_text())
    clean = json.loads((REPORT / "cleaning.json").read_text())
    c = proto["counts"]
    dep = res[DEPLOYED]
    gl = dep["galleries"]
    big = gl[-1]
    v0 = big["variants"]["single"]
    N, Nid = v0["n_nonmated_searches"], v0["n_nonmated_identities"]
    rec = recommend(DEPLOYED, dep)
    real_g = max((g for g in gl if not g["subjects_by_dataset"].get("digiface1m")), key=lambda g: g["subjects"])
    rr = rec["rows"][rec["margin"]]
    g3 = op(DEPLOYED, rec["D"], "group3", rec["t_id"], rec["margin"])
    tail = real_tail(DEPLOYED)
    ext = dep["extrapolation_1M_x4"]
    ext3 = op(DEPLOYED, rec["D"], "group3", ext["threshold"], 0.0)
    t_alt = 0.50
    alt, alt3 = op(DEPLOYED, rec["D"], "single", t_alt, rec["margin"]), op(DEPLOYED, rec["D"], "group3", t_alt, rec["margin"])
    L = ["# E2 — 1:N 오픈셋 식별 (FNIR@FPIR, 갤러리 확장, 등록 고객 오식별)", "",
         f"생성 {dep['created']} · `python -m eval.onen.run` (프로토콜 `{proto['name']}`, seed {proto['seed']}, "
         f"images.csv sha256 `{proto['images_csv_sha256'][:12]}…`). 상세 수치·CI·DET: `reports/E2_1N/<model>/results.json`, "
         "프로토콜/정리: `reports/E2_1N/protocol.json`, `cleaning.json`. 파이프라인 = 엔진과 동일 (SCRFD-34G 5점 → ArcFace "
         "norm_crop 112 → MODEL.json 정밀도, flip 없음), 검색 = 전수 내적 + 대상자별 템플릿 최대값, 판정 = top1 ≥ T AND "
         "top1−top2 ≥ margin. FPIR은 주 풀(CelebA 미등록 신원, 성인·신원 라벨 있음)로 산출, FFHQ probe는 보조 풀.", ""]
    rank_meas = sorted(res, key=lambda m: grid(res[m])[1e-4]["fnir"])
    rank_ext = sorted(res, key=lambda m: res[m]["extrapolation_1M_x4"]["fnir"]["0.0"])
    L += ["## 결론", "",
          f"- **권장값 (배포 모델 {DEPLOYED}): `FA_T_ID={rec['t_id']:.2f}`, `FA_MARGIN={rec['margin']:.2f}`** — 목표 "
          f"'FPIR 1e-6 @ 100만 명×4 템플릿'의 외삽 임계값 {rec['t_ext']:.3f} (GPD 적합 선택 범위 {rec['t_ext_range'][0]:.3f}–"
          f"{rec['t_ext_range'][1]:.3f})과 최대 갤러리 실측 FPIR {rec['fpir_meas']:.0e} 임계값 {rec['t_meas']:.3f} 중 큰 값을 "
          f"올림; margin은 T와 FPIR 1e-3 운영점 모두에서 FNIR 손실 ≤ {100 * MARGIN_COST:.1f}%p인 최대값.",
          f"- 그 운영점의 최대 갤러리 ({big['subjects']:,}명 / {big['templates']:,} 템플릿) 실측: FP {rr['fp']}/{rr['n_nonmated']:,} "
          f"(95% 상한 {rr['fpir_upper95']:.1e}); FNIR 단일 사진 **{pct(rr['fnir'])}%** {ci(rr['fnir_ci95'])}, 같은 사람 사진 "
          f"3장 평균 **{pct(g3['fnir'])}%** {ci(g3['fnir_ci95'])}; 등록 고객 타인 오식별 {rr['misid']}건.",
          f"- **RFP 목표(같은 임계값에서 FPIR ≤ 1e-6, FNIR ≤ 1%)는 이 데이터로 충족·입증되지 않는다.** FPIR: 독립 비동일인 "
          f"검색 {N:,}회 (신원 {Nid:,}명)로는 무오류여도 95% 상한 {v0['min_demonstrable_fpir_95']['by_search']:.1e} (사진 기준) / "
          f"{v0['min_demonstrable_fpir_95']['by_identity']:.1e} (신원 기준)까지만 입증된다 (1e-6에는 무오류 독립 검색 "
          f"{v0['searches_needed_for_fpir_1e-6']:,}회 필요). FNIR: 외삽 임계값 {ext['threshold']:.3f}에서 최대 갤러리 실측 "
          f"단일 사진 {pct(op(DEPLOYED, rec['D'], 'single', ext['threshold'], 0.0)['fnir'])}%, 3장 평균 {pct(ext3['fnir'])}% "
          f"(100만 명 모델 FNIR 단일 {pct(ext['fnir']['0.0'])}%) — 정답 점수가 임계값 아래인 어려운 사진(측면·노화·저화질)이 "
          "원인이다. 키오스크의 협조적 정면 다중 프레임에서의 FNIR은 현장 데이터로 다시 측정해야 한다.",
          f"- 대안 운영점 T={t_alt:.2f} (margin {rec['margin']:.2f}): 최대 갤러리 실측 FP {alt['fp']}/{alt['n_nonmated']:,}, "
          f"FNIR 단일 {pct(alt['fnir'])}% / 3장 평균 {pct(alt3['fnir'])}%, 외삽 FPIR @100만×4 ≈ "
          f"{metrics.fpir_model(tail.p(t_alt), N_TPL):.1e} (모델 기반) — 3장 평균이면 FNIR 1% 이내지만 100만 명 FPIR은 "
          "1e-6보다 수 자릿수 크다. FPIR 목표와 FNIR의 교환은 표 D.",
          "- **1:N 모델 순위**: 실측 (최대 갤러리 FNIR@FPIR 1e-4) " + " > ".join(
              f"{m} {pct(grid(res[m])[1e-4]['fnir'])}%" for m in rank_meas) + "; 외삽 (FPIR 1e-6@100만×4의 FNIR) " +
          " > ".join(f"{m} {pct(res[m]['extrapolation_1M_x4']['fnir']['0.0'])}%" for m in rank_ext) +
          f". 실측 1위 {rank_meas[0]}, 외삽 1위 {rank_ext[0]}; 배포 모델 {DEPLOYED}의 외삽 임계값이 높은 것은 비교당 타인 "
          "점수 꼬리가 더 길기 때문이다 (표 E의 T). R100(Glint360K)은 특정 유명인 쌍을 동일인처럼 묶는 오류(예: Katy Perry "
          "사진 → Zooey Deschanel 등록자 0.65–0.75, 다른 3개 모델은 같은 쌍에 0.05–0.29)로 FPIR ≤ 1e-4 영역이 무너진다.",
          f"- 데이터 정리 (필수였음): CelebA 신원 라벨 노이즈(한 사람이 여러 id, 한 id에 여러 사람, 같은 사진 중복)와 FFHQ의 유명인·"
          f"계정 간 반복 인물 때문에 비동일인 probe의 합의 점수 분포에 동일인 덩어리(0.5–0.8)가 있었고, 이것이 FPIR ≤ 1e-3 "
          f"임계값을 결정했다. 4개 모델 합의(평균 cos)로 CelebA id {len(clean['celeba_identities_dropped'])}개, "
          f"FFHQ 유명인 사진 {clean['ffhq_celebrity_dropped']}장, probe "
          f"{sum(clean['probes_dropped'].values()):,}장 제외; FFHQ는 Flickr 계정 단위로 probe/방해자 분할 (표 G).", ""]

    nm = v0["nonmated_by_dataset"]
    L += ["## 규모", "",
          "| 항목 | 값 |", "|---|---|",
          f"| 등록 대상자 (CelebA in-the-wild, 템플릿 ≤4장) | {c['enrolled_subjects']:,}명 / {c['template']['images']:,} 템플릿 |",
          f"| mated 검색 (등록자의 다른 사진) | {v0['n_mated_searches']:,}회 ({v0['n_mated_subjects']:,}명) |",
          f"| non-mated 검색 — 주 풀 (FPIR 산출) | CelebA {N:,}회 / 신원 {Nid:,}명 |",
          f"| non-mated 검색 — 보조 풀 | FFHQ {nm.get('ffhq', {}).get('searches', 0):,}회 (사진당 1명 가정; 영유아 포함) |",
          "| 방해 갤러리 | FFHQ " + f"{c['distractor']['by_dataset'].get('ffhq', {}).get('subjects', 0):,}명 (실사진, 1 템플릿), "
          f"DigiFace-1M {c['distractor']['by_dataset'].get('digiface1m', {}).get('subjects', 0):,}명 (합성, 4 템플릿) |",
          "| 갤러리 크기 | " + ", ".join(f"{g['subjects']:,}명/{g['templates']:,}T" for g in gl) + " |",
          "| 검출 실패 (모델 지표에서 제외) | " + ", ".join(f"{k} {x['n'] - x['ok']:,}/{x['n']:,} ({pct(x['fta_rate'])}%)"
                                               for k, x in proto["failure_to_acquire"].items()) + " |",
          f"| 3장 평균 probe 변형 | mated {big['variants']['group3']['n_mated_searches']:,}, "
          f"non-mated {big['variants']['group3']['n_nonmated_searches']:,} 그룹 (CelebA만) |", ""]

    L += [f"## A. 모델 비교 — 최대 갤러리 ({big['subjects']:,}명), 단일 사진, margin 0: FNIR % [Clopper-Pearson 95%] "
          "(부트스트랩 95%: 등록자·비동일인 신원 재표집, 임계값 재추정), 임계값", "",
          "| 모델 | rank-1 % | FPIR 1e-2 | FPIR 1e-3 | FPIR 1e-4 | FPIR 1e-5 |", "|---|---|---|---|---|---|"]
    for m, r in res.items():
        g = grid(r)
        cells = [f"{pct(g[f]['fnir'])} {ci(g[f]['fnir_ci95'])} ({ci(g[f]['fnir_at_fpir_boot_ci95'])}) t={g[f]['threshold']:.3f}"
                 + ("" if g[f]["resolvable"] else " ⚠") for f in (1e-2, 1e-3, 1e-4, 1e-5)]
        L.append(f"| {m} ({TRAIN.get(m, '')}) | {pct(r['galleries'][-1]['variants']['single']['rank1_identification_rate'])} | "
                 + " | ".join(cells) + " |")
    L += ["", f"⚠ = 해상도 한계 (FPIR 1e-5 × {N:,} = {1e-5 * N:.2f} → 허용 FP 0건). 신원 기준으로는 FPIR "
          f"{1 / Nid:.0e} 미만이 해상도 밖.", "",
          "보조 풀 FFHQ (영유아·계정 간 동일인 포함) — 최대 갤러리 FNIR % (FFHQ 풀 자체 임계값), 그리고 주 풀 FPIR 1e-4 "
          "임계값에서 FFHQ probe가 낸 FP:", "",
          "| 모델 | FPIR 1e-3 | FPIR 1e-4 | 주 풀 1e-4 임계값에서 FP: CelebA / FFHQ |", "|---|---|---|---|"]
    n_ff = nm.get("ffhq", {}).get("searches", 0)
    for m, r in res.items():
        g, gp = grid(r, pool="ffhq"), grid(r)[1e-4]
        fp = gp["fp_by_probe_dataset"]
        L.append(f"| {m} | {pct(g[1e-3]['fnir'])} ({g[1e-3]['threshold']:.3f}) | {pct(g[1e-4]['fnir'])} ({g[1e-4]['threshold']:.3f}) "
                 f"| {fp.get('celeba', 0)}/{N:,} / {fp.get('ffhq', 0)}/{n_ff:,} |")
    L += ["", f"{DEPLOYED} 전체 풀 상위 16건 육안 확인: 영유아 대 영유아 8건, 나머지는 계정이 다른 동일인으로 보이는 쌍이 "
          "대부분 (results.json `top_nonmated_all`). 고객(성인) 대상 FPIR에는 주 풀을 쓴다.", ""]

    L += ["## B. 갤러리 확장 — FNIR % @ FPIR (임계값), 단일 사진, margin 0", "",
          "| 모델 | FPIR | " + " | ".join(f"{g['subjects']:,}명" for g in gl) + " |", "|---|---|" + "---|" * len(gl)]
    for m, r in res.items():
        for f in (1e-3, 1e-4):
            L.append(f"| {m} | {f:.0e} | " + " | ".join(
                f"{pct(grid(r, i)[f]['fnir'])} ({grid(r, i)[f]['threshold']:.3f})" for i in range(len(r["galleries"]))) + " |")
    L += ["", f"실사진 갤러리는 {real_g['subjects']:,}명까지 (CelebA 등록자 + FFHQ), 그 이상은 DigiFace 합성 방해자. 주 풀(성인 "
          "유명인)에 대해 FFHQ·DigiFace 방해자는 거의 오수락을 만들지 않아 임계값이 거의 변하지 않는다 (표 F의 출처별 꼬리).", ""]

    L += [f"## C. margin 규칙 ({DEPLOYED}, 최대 갤러리, 단일 사진)", "",
          "| margin | FNIR@FPIR 1e-3 (t) | FNIR@FPIR 1e-4 (t) | 오식별@1e-4 | " + f"T={rec['t_id']:.2f}: FNIR / FP / 오식별 |",
          "|---|---|---|---|---|"]
    for mg in MARGINS:
        g = grid(dep, m=str(mg))
        o = rec["rows"][mg]
        L.append(f"| {mg:.2f} | {pct(g[1e-3]['fnir'])} ({g[1e-3]['threshold']:.3f}) | {pct(g[1e-4]['fnir'])} "
                 f"({g[1e-4]['threshold']:.3f}) | {g[1e-4]['misid']} | {pct(o['fnir'])} / {o['fp']} / {o['misid']} |")
    L.append("")

    L += [f"## D. 임계값 교환표 ({DEPLOYED}, margin {rec['margin']:.2f}; 실측 = 최대 갤러리 {big['subjects']:,}명)", "",
          "| T | 외삽 FPIR @100만×4 (모델 기반) | 실측 FP / non-mated (95% 상한) | FNIR 단일 % [CP 95%] | FNIR 3장 평균 % | 오식별 건 |",
          "|---|---|---|---|---|---|"]
    for t in sorted({0.40, 0.45, 0.50, 0.55, 0.60, round(ext["threshold"], 3), rec["t_id"]}):
        o, o3 = op(DEPLOYED, rec["D"], "single", t, rec["margin"]), op(DEPLOYED, rec["D"], "group3", t, rec["margin"])
        L.append(f"| {t:.3f} | {metrics.fpir_model(tail.p(t), N_TPL):.1e} | {o['fp']} / {o['n_nonmated']:,} ({o['fpir_upper95']:.1e}) "
                 f"| {pct(o['fnir'])} {ci(o['fnir_ci95'])} | {pct(o3['fnir'])} | {o['misid']} |")
    L += ["", f"단일 사진 FNIR 하한 ~{pct(1 - v0['rank1_identification_rate'], 1)}%는 rank-1 실패(측면·가림·저화질·잔여 라벨 "
          "노이즈)이며 임계값과 무관하다.", ""]
    L += [f"## E. 1M명×4 템플릿 외삽 (모델 기반 추정, 측정 아님)", "",
          "비교당 오수락 확률 p(t) = 주 풀 probe × 실인물 템플릿(CelebA 등록 + FFHQ) 전 비교의 점수 분포 "
          f"({tail.n:.2e}회, 경험적 해상도 {1 / tail.n:.0e}); 그 아래는 GPD 꼬리. 독립 가정: FPIR(T) = 1 − (1 − p(T))^4M → "
          "p* = 2.5e-13. FNIR = 정답 점수가 T 미만이거나 4M개 타인 템플릿 중 하나라도 정답 − margin 이상.", "",
          "| 모델 | T (GPD 적합 범위) | GPD shape | FNIR 단일 @T | 실측 FNIR@T (최대 갤러리) | 3장 평균 @T | 독립모델 검증: 예측/실측 FPIR |",
          "|---|---|---|---|---|---|---|"]
    for m, r in res.items():
        e = r["extrapolation_1M_x4"]
        sens = [x["threshold"] for x in e["gpd_threshold_sensitivity"]]
        shape = r["impostor_tails"]["real (celeba+ffhq templates)"]["gpd_shape"]
        D = r["galleries"][-1]["distractor_subjects"]
        meas, meas3 = op(m, D, "single", e["threshold"], 0.0), op(m, D, "group3", e["threshold"], 0.0)
        chk = [x for x in r["model_check"]["rows"] if x["fpir_measured"] > 0]
        ratio = (f"{min(x['fpir_model'] / x['fpir_measured'] for x in chk):.2f}–"
                 f"{max(x['fpir_model'] / x['fpir_measured'] for x in chk):.2f}") if chk else "—"
        tcell = (f"{e['threshold']:.3f} ({min(sens):.3f}–{max(sens):.3f})" if max(sens) <= 1 else
                 f"{e['threshold']:.3f} ({min(sens):.3f}–>1: 꼬리가 무거워 결정 불가)")
        L.append(f"| {m} | {tcell} | {shape:+.3f} | {pct(e['fnir']['0.0'])} | {pct(meas['fnir'])} | {pct(meas3['fnir'])} | {ratio} |")
    L += ["", "예측/실측 FPIR (FP ≥ 3건인 측정점들): 1 근처면 독립 모형이 측정과 맞는다. 배포 모델의 경험 해상도 끝 "
          f"(0.50)에서 GPD p = {tail.p(0.5):.1e} vs 경험값 {tail.p_emp(0.5):.1e} (fit_check: results.json "
          f"`impostor_tails`). p* = 2.5e-13은 경험 해상도({1 / tail.n:.0e})보다 약 {math.log10(1 / tail.n / 2.5e-13):.1f} 자릿수 "
          "아래이므로 T는 GPD 적합 범위 정도, FPIR은 몇 배 수준의 불확실성을 갖는다.", ""]

    L += ["## F. 3장 평균 probe, 방해자 출처별 꼬리", "",
          "| 모델 | 3장 평균 FNIR@FPIR 1e-3 | @1e-4 | 비교당 P(score ≥ 0.30): CelebA 템플릿 / FFHQ / DigiFace |", "|---|---|---|---|"]
    for m, r in res.items():
        g = grid(r, v="group3")
        p = r["impostor_tails"]["p_at"]["0.30"]
        L.append(f"| {m} | {pct(g[1e-3]['fnir'])} | {pct(g[1e-4]['fnir'])} | {p['celeba']:.1e} / {p['ffhq']:.1e} / "
                 f"{p['digiface1m']:.1e} |")
    L += ["", "주 풀(성인 유명인 probe) 기준으로 DigiFace(합성)·FFHQ 템플릿과의 타인 점수는 CelebA 템플릿보다 훨씬 낮다 → 방해자로 "
          "갤러리를 키워도 실제 고객 갤러리만큼 어려워지지 않는다. 외삽은 실인물 꼬리만 쓴다.", ""]

    pd_ = clean["probes_dropped"]
    L += ["## G. 데이터 정리 (4개 모델 합의, `reports/E2_1N/cleaning.json`)", "",
          f"- CelebA: 신원 중심(centroid) 합의 cos ≥ {clean['rules']['tau_identity']} 인 id 쌍 {clean['duplicate_pairs']}개 → "
          f"성분마다 가장 큰 id만 남기고 {len(clean['celeba_identities_dropped'])}개 id 제외.",
          f"- FFHQ: CelebA 신원 중심과 합의 cos ≥ {clean['rules']['tau_ffhq_celebrity']} 인 사진 {clean['ffhq_celebrity_dropped']}장 "
          "(유명인) 제외; probe/방해자는 Flickr 계정 단위로 분할.",
          "- probe: 자기 대상자를 뺀 최고 합의 점수가 같은 출처 갤러리 대상자에 대해 임계 이상이면 제외 — " +
          ", ".join(f"{k} {v:,}" for k, v in pd_.items() if v) + " (임계: " +
          ", ".join(f"{k} {v}" for k, v in clean["rules"]["tau_probe"].items()) + "; 합의 점수 히스토그램의 타인 덩어리와 "
          "동일인 덩어리 사이 골짜기).",
          "- 육안 점검: 제외된 CelebA probe 무작위 32장은 대부분 일치한 대상자의 템플릿 중 하나와 같은 사람이었다 (일부는 판단 "
          "보류). 정리는 모든 모델이 동의하는 경우만 제거하므로, 한 모델만 높게 매기는 쌍(예: R100의 Katy Perry–Zooey "
          "Deschanel)은 그 모델의 오류로 남는다. 반대로 4개 모델이 모두 혼동하는 실제 닮은꼴도 제거됐을 수 있다 (FPIR "
          "낙관 편향, 규모는 정리 전후 차이 이하).", ""]

    L += ["## 한계", "",
          f"- 규모: 최대 갤러리 {big['subjects']:,}명 (실인물 {real_g['subjects']:,}명) vs 목표 100만 명; FPIR 1e-6은 외삽으로만 "
          "제시 (GPD 꼬리 + 비교 독립 가정).",
          "- 신원 중복: CelebA(유명인)는 WebFace/Glint 학습 신원과 겹칠 가능성이 높아 낙관적이다 (R100의 유명인 혼동처럼 "
          "학습 데이터 라벨 오류도 그대로 드러남). FFHQ는 비유명인이지만 사진당 1명 가정이 보장되지 않는다. DigiFace는 합성이다.",
          "- 동아시아 전용 세트 없음: InsightFace MR-ALL에서 공개 모델은 East Asian 정확도가 다른 집단보다 크게 낮다 "
          "(buffalo_l 74.96 vs Caucasian 94.70). 한국 고객 대상 운영 전 동아시아 1:N 재측정 필요.",
          "- 정지 사진(웹/Flickr) ≠ 키오스크 촬영: 실제 템플릿은 안내된 다방향 프레임 평균, probe는 같은 세션의 인접 프레임 "
          "평균이다. 3장 평균 변형은 서로 다른 사진이라 프레임 평균과 성격이 다르다.",
          "- 라벨 노이즈 잔여분과 정리의 모델 의존성 (표 G). 정리 기준은 결과를 본 뒤 정했다 (히스토그램 골짜기) — "
          "임계·운영점 결정 전에 고정된 독립 시험셋으로 재확인 필요 (RFP §5).",
          "- 인구 구성: 주 풀은 성인 유명인이다. 같은 임계값에서 FFHQ(일반인, 영유아 포함) probe의 FP 비율은 수십 배 높다 "
          "(표 A 아래). 영유아·계정 간 동일인이 섞여 있어 실제 오수락 비율은 미확정이지만, 고객 인구가 넓을수록 FPIR이 "
          "나빠질 수 있다는 경고로 읽어야 한다."]
    (REPORT / "summary.md").write_text("\n".join(L) + "\n")
    return rec


def e6_summary():
    r = json.loads((REPORT_E6 / "results.json").read_text())
    L = ["# E6 — 정렬 ablation: SCRFD 5점 vs FaRL 유도 5점 (1:N / 1:1)", "",
         f"생성 {r['created']} · `python -m eval.onen.run e6-*`. E2 프로토콜 `{r['protocol']}`의 CelebA 부분집합 "
         f"(seed {r['subset']['seed']}): 등록 {r['subset']['enrolled']:,}명 (템플릿 {r['counts']['template']:,}), "
         f"mated {r['counts']['mated']:,}회, 미등록 {r['subset']['nonenrolled']:,}명 non-mated {r['counts']['nonmated']:,}회. "
         "갤러리 = 부분집합 등록자만. 정렬만 다르고 검출·이미지·프로토콜은 동일.", "",
         f"- scrfd: {r['alignments']['scrfd']}", f"- farl: {r['alignments']['farl']}", ""]
    if r.get("farl_align"):
        L += [f"FaRL 정렬 실패: {r['farl_align']['n'] - r['farl_align']['ok']} / {r['farl_align']['n']}", ""]
    L += ["| 모델 | 정렬 | FNIR@FPIR 1e-2 | 1e-3 | 1e-4 | rank-1 | TAR@FAR 1e-4 | 1e-5 | 1e-6 |", "|---|---|---|---|---|---|---|---|---|"]
    for m, d in r["models"].items():
        for al in ("scrfd", "farl"):
            ev = d[al]
            g = {row["target_fpir"]: row for row in ev["margins"]["0.0"]["grid"]}
            t = {x["far_target"]: x for x in ev["one_to_one"]}
            L.append(f"| {m} | {al} | " + " | ".join(f"{pct(g[f]['fnir'])} {ci(g[f]['fnir_ci95'])}" for f in (1e-2, 1e-3, 1e-4))
                     + f" | {pct(ev['rank1_identification_rate'])} | " +
                     " | ".join(f"{pct(t[f]['tar'])}" if f in t else "—" for f in (1e-4, 1e-5, 1e-6)) + " |")
    L += ["", "같은 mated probe 쌍비교 (각 정렬의 자체 임계값, margin 0): SCRFD만 성공 / FaRL만 성공 (exact McNemar p)", ""]
    for m, d in r["models"].items():
        L.append(f"- {m}: " + "; ".join(f"FPIR {float(f):.0e}: {x['hit_scrfd_only']} / {x['hit_farl_only']} (p={x['p_exact']:.2g})"
                                        for f, x in d["paired_mcnemar"].items()))
    nm = r["counts"]["nonmated"]
    L += ["", f"non-mated {nm:,}회 → FPIR 1e-4 = 허용 FP {int(1e-4 * nm)}건 (해상도 밖, 비교는 1e-2·1e-3과 쌍비교 위주).", "",
          "**결론**: 두 모델 모두 FaRL 유도 5점 정렬의 FNIR이 SCRFD 5점보다 나쁘고 (같은 probe 쌍비교에서 SCRFD만 성공한 "
          "probe가 유의하게 많음), 1:1 TAR 차이는 0.2%p 내외다. 인식 모델이 RetinaFace/SCRFD 계열 5점으로 정렬된 데이터로 "
          "학습됐고 FaRL 유도 입꼬리 위치가 SCRFD와 체계적으로 다르기 때문으로 보인다 (E5 `scrfd_consistency`: 입꼬리 평균 "
          "편차 0.012–0.024 IOD). 인식 정렬은 SCRFD kps를 유지하고 FaRL은 랜드마크·품질 용도로만 쓴다 (설계 §4 재검증).", ""]
    (REPORT_E6 / "summary.md").write_text("\n".join(L) + "\n")


def main():
    if (REPORT / "protocol.json").exists() and (REPORT / DEPLOYED / "results.json").exists():
        rec = e2_summary()
        print(f"E2 summary written; FA_T_ID={rec['t_id']:.2f} FA_MARGIN={rec['margin']:.2f}")
    if (REPORT_E6 / "results.json").exists():
        e6_summary()
        print("E6 summary written")
