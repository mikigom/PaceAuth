// Kiosk screen renderer shared by kiosk.html (real-time replay) and video.html (frame-exact video).
// Draws one state of the kiosk; the state comes from the run recorded on the real stack (scripts/demo_record.py ->
// data/demo/timeline.js). The kiosk only ever shows what the device receives: prompts from the terminal guide /
// trigger and the external decision (scores and internal reasons stay on the server).
(function () {
  "use strict";
  const TICKS = 60, FW = 640, FH = 480, FRAMES = "../data/demo/frames/";
  const PROMPT_CENTRE = "정면을 바라봐 주세요";
  const NAMES = { A: "사용자 A", B: "사용자 B", C: "사용자 C", D: "사용자 D" };
  const HINTS = { idle: "화면 가까이 오시면 자동으로 인식합니다", far: "조금 더 가까이 와 주세요",
    too_close: "조금 뒤로 물러나 주세요", adjust: "정면을 바라봐 주세요", multi: "한 분만 화면에 들어와 주세요",
    ready: "인식 중입니다" };
  const AMR = { face: "얼굴", otp: "OTP", pin: "PIN" };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const ease = (x) => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const pending = [];

  // ------------------------------------------------------------------ DOM
  function build(root) {
    root.classList.add("kiosk");
    root.innerHTML = `
      <div class="k-bar"><span class="k-clock">09:41</span><span class="k-dev"></span><span class="k-net">● 보안 연결</span></div>
      <div class="k-head"><div class="k-brand">FaceAuth</div><div class="k-title"></div></div>
      <div class="k-cam">
        <div class="k-view"><div class="k-mirror"><img class="k-img" alt=""></div><div class="k-shade"></div></div>
        <svg class="k-ring" viewBox="-100 -100 200 200"></svg>
        <div class="k-badge"></div>
      </div>
      <div class="k-prompt"></div>
      <div class="k-sub"></div>
      <div class="k-dist"><div class="k-dist-track"><div class="k-dist-band"></div><div class="k-dist-dot"></div></div>
        <div class="k-dist-label"></div></div>
      <div class="k-actions"></div>
      <div class="k-home"></div>`;
    const svg = root.querySelector(".k-ring");
    const ticks = [];
    for (let i = 0; i < TICKS; i++) {
      const l = document.createElementNS("http://www.w3.org/2000/svg", "line");
      svg.appendChild(l);
      ticks.push(l);
    }
    const q = (s) => root.querySelector(s);
    return { root, ticks, img: q(".k-img"), mirror: q(".k-mirror"), view: q(".k-view"), shade: q(".k-shade"),
      badge: q(".k-badge"), prompt: q(".k-prompt"), sub: q(".k-sub"), title: q(".k-title"), dev: q(".k-dev"),
      dist: q(".k-dist"), distDot: q(".k-dist-dot"), distLabel: q(".k-dist-label"), actions: q(".k-actions"),
      home: q(".k-home"), last: {} };
  }

  // tick i covers head-direction azimuth [6i, 6i+6) (0 = the subject's left, 90 = up; terminal/guide.py
  // ring_position); the preview is mirrored, so the subject's left is drawn on the screen's left.
  function tickLine(l, i, r0, r1) {
    const phi = (180 - (i + 0.5) * 6) * Math.PI / 180, c = Math.cos(phi), s = -Math.sin(phi);
    l.setAttribute("x1", (r0 * c).toFixed(2)); l.setAttribute("y1", (r0 * s).toFixed(2));
    l.setAttribute("x2", (r1 * c).toFixed(2)); l.setAttribute("y2", (r1 * s).toFixed(2));
  }

  function setText(k, key, node, text) {
    if (k.last[key] !== text) { node.innerHTML = text; k.last[key] = text; }
  }

  function render(k, s) {
    setText(k, "title", k.title, s.title || "");
    setText(k, "dev", k.dev, s.device || "");
    // camera: crop (cx, cy, size) of the 640x480 frame into the circle, mirrored
    const D = k.view.clientWidth || 380;
    if (s.frame) {
      const src = FRAMES + s.frame + "_rgb.jpg";
      if (k.last.src !== src) {
        k.img.src = src; k.last.src = src;
        if (k.img.decode) pending.push(k.img.decode().catch(() => {}));
      }
      const c = s.crop || { cx: FW / 2, cy: FH / 2, size: FH }, sc = D / c.size;
      k.img.style.width = (FW * sc) + "px"; k.img.style.height = (FH * sc) + "px";
      k.img.style.left = (D / 2 - c.cx * sc) + "px"; k.img.style.top = (D / 2 - c.cy * sc) + "px";
      k.img.style.opacity = 1;
    } else {
      k.img.style.opacity = 0;
    }
    k.shade.style.opacity = s.dim || 0;
    // ring
    const r = s.ring || {};
    const lit = r.lit instanceof Set ? r.lit : new Set(r.lit || []);
    for (let i = 0; i < TICKS; i++) {
      const on = r.all || lit.has(i);
      let len = on ? 1 : 0, col = on ? (r.color || "#30d158") : "#4a4a50", op = on ? 1 : (r.dimOpacity ?? 0.9);
      if (r.spin != null) {                           // recognising: a comet running around the ring
        const d = ((i - r.spin) % TICKS + TICKS) % TICKS;
        const w = Math.max(0, 1 - d / 18);
        len = w; col = "#0a84ff"; op = 0.25 + 0.75 * w;
      }
      if (r.level != null && !on && r.spin == null) { op = 0.2 + 0.8 * r.level; col = r.levelColor || "#8e8e93"; }
      const key = `${len.toFixed(2)}|${col}|${op.toFixed(2)}`;
      if (k.last["t" + i] !== key) {
        tickLine(k.ticks[i], i, 84 - 6 * len, 95 + 2 * len);
        k.ticks[i].setAttribute("stroke", col); k.ticks[i].setAttribute("stroke-opacity", op.toFixed(2));
        k.ticks[i].setAttribute("stroke-width", "2.6"); k.ticks[i].setAttribute("stroke-linecap", "round");
        k.last["t" + i] = key;
      }
    }
    // badge
    const badge = s.badge ? `<div class="kb kb-${s.badge}" style="transform:scale(${(0.6 + 0.4 * ease(s.badgeT ?? 1)).toFixed(3)});opacity:${ease((s.badgeT ?? 1) * 2).toFixed(3)}">${
      s.badge === "check" ? '<svg viewBox="0 0 52 52"><path d="M14 27 l8 8 l16 -18"/></svg>'
      : s.badge === "cross" ? '<svg viewBox="0 0 52 52"><path d="M17 17 L35 35 M35 17 L17 35"/></svg>'
      : '<svg viewBox="0 0 52 52"><path d="M26 14 v14 M26 36 v2"/></svg>'}</div>` : "";
    setText(k, "badge", k.badge, badge);
    setText(k, "prompt", k.prompt, s.prompt || "");
    setText(k, "sub", k.sub, s.sub || "");
    // distance gauge (depth-measured), 0.3-1.6 m with the 0.4-0.8 m capture band
    if (s.dist) {
      k.dist.style.opacity = 1;
      const x = s.dist.mm == null ? null : clamp((s.dist.mm - 300) / 1300, 0, 1);
      k.distDot.style.opacity = x == null ? 0 : 1;
      if (x != null) k.distDot.style.left = (x * 100).toFixed(2) + "%";
      k.distDot.classList.toggle("in", s.dist.mm != null && s.dist.mm >= 400 && s.dist.mm <= 800);
      setText(k, "dl", k.distLabel, s.dist.mm == null ? "거리 측정 대기" : `거리 ${(s.dist.mm / 1000).toFixed(2)} m · 깊이 센서 측정`);
    } else {
      k.dist.style.opacity = 0;
    }
    setText(k, "actions", k.actions, (s.actions || []).map((a) => `<div class="k-btn">${a}</div>`).join(""));
    // logged-in home screen sliding up over the camera screen
    if (s.home) {
      const h = s.home, p = ease(h.t);
      k.home.style.transform = `translateY(${((1 - p) * 100).toFixed(2)}%)`;
      k.home.style.opacity = 1;
      setText(k, "home", k.home, `
        <div class="kh-top"><div class="kh-hello">안녕하세요,</div><div class="kh-name">${h.name}님</div>
        <div class="kh-chip">얼굴 인증으로 로그인됨</div></div>
        <div class="kh-card"><div class="kh-label">계정</div><div class="kh-val">${h.account}</div></div>
        <div class="kh-card"><div class="kh-label">인증 방법</div><div class="kh-val">${h.amr}</div></div>
        <div class="kh-card"><div class="kh-label">승인 토큰</div><div class="kh-val">Ed25519 서명 검증 · ${h.ttl}초 만료</div></div>
        <div class="kh-grid"><div>내 정보</div><div>이용 내역</div><div>설정</div><div>로그아웃</div></div>`);
    } else {
      k.home.style.opacity = 0; k.home.style.transform = "translateY(100%)";
    }
  }

  // ------------------------------------------------------------------ timeline -> state
  function bisect(ticks, t) {                       // last index with ticks[i].t <= t (0 if none)
    let lo = 0, hi = ticks.length - 1;
    if (t < ticks[0].t) return 0;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ticks[m].t <= t) lo = m; else hi = m - 1; }
    return lo;
  }

  // smoothed crop per tick: follows the face, clamped so a far face stays small (the person is far away)
  function cropTrack(seg, alpha, minSize) {
    let c = { cx: FW / 2, cy: FH / 2, size: FH };
    return seg.ticks.map((tk) => {
      if (tk.bbox) {
        const [x1, y1, x2, y2] = tk.bbox, w = Math.max(x2 - x1, y2 - y1);
        const g = { cx: (x1 + x2) / 2, cy: (y1 + y2) / 2 + 0.06 * w, size: clamp(2.3 * w, minSize, FH) };
        c = { cx: c.cx + alpha * (g.cx - c.cx), cy: c.cy + alpha * (g.cy - c.cy), size: c.size + alpha * (g.size - c.size) };
      }
      return c;
    });
  }

  const DIRS = ["C", "L", "UL", "U", "UR", "R", "DR", "D", "DL"];

  // enrollment: ts = sensor seconds since the first prompt
  function enrollState(seg, ts, device) {
    seg._crop = seg._crop || cropTrack(seg, 0.45, 200);
    const ticks = seg.ticks, end = ticks[ticks.length - 1].t, lat = Math.max(seg.server.latency_s, 0.9);
    const i = bisect(ticks, ts), tk = ticks[i];
    const base = { title: "얼굴 등록", device, frame: tk.frame, crop: seg._crop[i] };
    const bins = DIRS.filter((d) => (tk.coverage[d] || 0) >= 2).length;
    if (ts <= end) {
      const circling = tk.prompt !== PROMPT_CENTRE && bins >= 1;
      return { ...base, ring: { lit: tk.ring }, prompt: tk.prompt,
        sub: circling ? `방향 ${bins} / 9 수집` : "얼굴을 원 안에 맞춰 주세요",
        actions: ["접근성 옵션"], phase: "capture", coverage: tk.coverage, bins };
    }
    if (ts <= end + lat) {
      return { ...base, ring: { all: true }, prompt: "등록 확인 중…", sub: "서버에서 9방향·품질·위조·중복 확인",
        badge: "spin", dim: 0.35, phase: "server", coverage: tk.coverage, bins: 9 };
    }
    return { ...base, ring: { all: true }, badge: "check", badgeT: (ts - end - lat) / 0.5, dim: 0.45,
      prompt: "등록 완료", sub: `${NAMES[seg.user] || seg.user}님의 얼굴이 등록되었습니다`, phase: "done",
      coverage: tk.coverage, bins: 9 };
  }

  // authentication: lt = seconds since the person started walking up (30 fps recording)
  function authState(seg, lt, device) {
    seg._crop = seg._crop || cropTrack(seg, 0.18, 220);
    const ticks = seg.ticks, sv = seg.server, fps = 30;
    const i = clamp(Math.floor(lt * fps), 0, ticks.length - 1), tk = ticks[i];
    const base = { title: "로그인", device, frame: tk.frame, crop: seg._crop[i] };
    const answer = sv.sent_t + sv.latency_s;
    if (lt < sv.trigger_t) {
      const mm = tk.distance_mm, level = mm == null ? 0 : clamp((1500 - mm) / 700, 0, 1);
      return { ...base, ring: { lit: [], level, levelColor: tk.state === "ready" ? "#0a84ff" : "#8e8e93" },
        prompt: HINTS[tk.state] || HINTS.idle, sub: tk.state === "idle" ? "버튼을 누를 필요가 없습니다" : "",
        dist: { mm }, phase: "approach", tick: tk };
    }
    if (lt < answer) {
      return { ...base, ring: { spin: ((lt - sv.trigger_t) * 70) % TICKS }, prompt: HINTS.ready,
        sub: "3프레임 서명 전송 · 서버 판정 중", dist: { mm: tk.distance_mm }, phase: "recognizing", tick: tk };
    }
    const rt = lt - answer, d = sv.response.decision, why = sv.response.reason;
    if (d === "approved") {
      const name = NAMES[seg.user] || seg.user, lg = sv.login || {};
      return { ...base, ring: { all: true }, badge: "check", badgeT: rt / 0.45, dim: 0.4,
        prompt: `${name}님, 환영합니다`, sub: "로그인되었습니다", phase: "approved", tick: tk,
        home: rt > 1.4 ? { t: (rt - 1.4) / 0.7, name, account: lg.account, ttl: lg.ttl_s,
          amr: (lg.amr || []).map((a) => AMR[a] || a).join(" + ") } : null };
    }
    const msg = why === "not_recognized" ? ["등록된 사용자를 찾을 수 없습니다", "다른 방법으로 로그인해 주세요", ["PIN으로 로그인", "휴대폰 인증"]]
      : why === "retry" ? ["다시 시도해 주세요", "화면을 정면으로 바라봐 주세요", []]
      : ["인증할 수 없습니다", "직원에게 문의해 주세요", ["직원 호출"]];
    return { ...base, ring: { all: true, color: "#ff453a" }, badge: "cross", badgeT: rt / 0.45, dim: 0.45,
      prompt: msg[0], sub: msg[1], actions: msg[2], phase: "rejected", tick: tk };
  }

  function settle() {                               // resolves when every image set so far is decoded
    const p = Promise.all(pending.splice(0));
    return p;
  }

  window.Kiosk = { build, render, enrollState, authState, settle, NAMES, DIRS, FRAMES, clamp, ease };
})();
