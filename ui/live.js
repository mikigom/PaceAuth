"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const views = {
    rgb: { label: "컬러", alt: "좌우 반전된 카메라 컬러 영상", description: "거울처럼 보이는 내 모습" },
    depth: { label: "깊이", alt: "좌우 반전된 카메라 깊이 영상", description: "색으로 보는 카메라와의 거리 · 좌우 반전" },
    ir: { label: "적외선", alt: "좌우 반전된 카메라 적외선 영상", description: "적외선으로 보는 내 모습 · 좌우 반전" },
  };
  const observationModes = {
    diagnostic: { label: "센서 진단", description: "정면에서 5개 프레임의 진단 점수를 수집합니다. 인증·등록 촬영의 운영 검증 표본과 구분됩니다." },
    auth: { label: "인증 촬영 검증", description: "인증과 같은 정면 촬영 절차로 판별 점수를 수집합니다. 얼굴 인증을 승인하지 않습니다." },
    enroll: { label: "등록 촬영 검증", description: "등록과 같은 9방향 촬영 절차로 판별 점수를 수집합니다. 얼굴을 등록하거나 계정에 얼굴 특징을 저장하지 않습니다." },
  };
  const image = $("preview-image");
  let selectedView = "rgb";
  let latestStatus = null;
  let statusBusy = false;
  let previewBusy = false;
  let previewGeneration = 0;
  let previewController = null;
  let previewTimer = null;
  let statusController = null;
  let lastSequence = null;
  let displayStartedAt = null;
  const displayTimes = [];
  let imageUrl = null;
  let frameDisplayed = false;
  let statusAt = 0;
  let authBusy = false;
  let operationBusy = false;
  let operationController = null;
  let currentOperation = null;
  let operationId = null;
  let dismissedOperationId = null;
  let operationUncertain = false;
  let consentRevision = null;
  let cancelRequested = false;
  let lastOperationKind = null;
  let pendingRequestId = null;
  let signedAccountId = null;
  let accountMode = "login";
  let profileAccountId = null;
  let profileDirty = false;
  let faceBusy = false;
  let faceController = null;
  let latestFaces = null;
  let facesReceivedAt = 0;
  let facesAgeAtReceipt = 0;
  let facesRenderKey = null;
  let displayedCaptureAt = null;
  let mainView = "live";
  let managedAccounts = [];
  let accountsLoaded = false;
  let accountsLoading = false;
  let accountsReady = false;
  let accountsMessage = "";
  let accountsOwner = null;
  let accountsFetchedAt = 0;
  let consolePickerKey = null;
  let selectedAccountId = null;
  let managedProfileDirty = false;
  let accountsRenderKey = null;
  let detailRenderKey = null;
  let enrollmentTarget = null;
  let refreshedOperationId = null;
  let deletionTarget = null;
  let deletionKind = "account";
  let enrollmentMode = false;
  let enrollmentRingKey = null;
  let guideReceivedAt = 0;
  let guideAgeAtReceipt = Infinity;
  let lastOperationPollAt = 0;
  let enrollmentPoseSequence = null;
  const MAX_GUIDE_AGE_MS = 500;
  const MAX_FACE_AGE_MS = 600;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const activeStates = new Set(["starting", "capturing", "processing", "cancelling"]);
  const enrollmentDirections = ["U", "UR", "R", "DR", "D", "DL", "L", "UL"];
  const enrollmentDirectionNames = { C: "정면", U: "위", UR: "오른쪽 위", R: "오른쪽", DR: "오른쪽 아래", D: "아래", DL: "왼쪽 아래", L: "왼쪽", UL: "왼쪽 위" };

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function badge(id, text, tone) {
    $(id).textContent = text;
    $(id).dataset.tone = tone;
  }

  function indicator(text, tone) {
    $("preview-label").textContent = text;
    $("preview-indicator").dataset.tone = tone;
  }

  function clearPreview(title, description) {
    previewGeneration += 1;
    previewController?.abort();
    clearTimeout(previewTimer);
    previewTimer = null;
    lastSequence = null;
    displayedCaptureAt = null;
    clearFaceOverlay();
    displayStartedAt = null;
    displayTimes.length = 0;
    $("display-fps").firstChild.textContent = "—";
    frameDisplayed = false;
    image.hidden = true;
    image.removeAttribute("src");
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = null;
    $("preview-kind").hidden = true;
    $("preview-placeholder").hidden = false;
    $("placeholder-title").textContent = title;
    $("placeholder-description").textContent = description;
    renderEnrollment();
  }

  function displayMetrics(now = performance.now()) {
    while (displayTimes.length && displayTimes[0] < now - 2000) displayTimes.shift();
    const elapsed = displayStartedAt === null ? 0 : now - displayStartedAt;
    const count = elapsed >= 2000 ? displayTimes.length : Math.max(0, displayTimes.length - 1);
    $("display-fps").firstChild.textContent = elapsed >= 1000 && frameDisplayed
      ? (count * 1000 / Math.min(2000, elapsed)).toFixed(1) : "—";
  }

  function metrics(capture = {}) {
    // Only values reported by the real capture service are shown.
    $("capture-fps").firstChild.textContent = finite(capture.fps) ? capture.fps.toFixed(1) : "—";
    $("depth-distance").firstChild.textContent = finite(capture.depth_median_mm) && capture.depth_median_mm > 0
      ? (capture.depth_median_mm / 1000).toFixed(2) : "—";
    $("detail-frames").textContent = finite(capture.frames) ? capture.frames.toLocaleString("ko-KR") : "—";
    $("detail-depth-valid").textContent = finite(capture.depth_valid_ratio)
      ? `${(capture.depth_valid_ratio * 100).toFixed(1)}%` : "—";
    $("frame-age").textContent = finite(capture.age_ms)
      ? `최근 수신 ${(Math.max(0, capture.age_ms) / 1000).toFixed(1)}초 전` : "프레임 대기";
  }

  function isStreaming() {
    const age = latestStatus?.capture?.age_ms;
    return latestStatus?.camera?.state === "streaming"
      && finite(age) && age < 3000 && Date.now() - statusAt < 4000;
  }

  function clearFaceOverlay() {
    $("face-overlay").replaceChildren();
    $("face-overlay").toggleAttribute("hidden", true);
    facesRenderKey = null;
  }

  function liveMessage(message) {
    if ($("live-recognition-message").textContent !== message) $("live-recognition-message").textContent = message;
  }

  function renderFaces(now = performance.now()) {
    const data = latestFaces;
    badge("live-recognition-state", "실시간 인식", data?.state === "running" ? "good" : "waiting");
    const age = facesAgeAtReceipt + now - facesReceivedAt;
    const fresh = data?.enabled === true && data.state === "running" && finite(age) && age < MAX_FACE_AGE_MS;
    $("recognition-fps").textContent = fresh && finite(data.fps) ? Math.max(0, data.fps).toFixed(1) : "—";
    if (!data || !fresh || document.hidden || !isStreaming() || !frameDisplayed || activeOperation() || operationUncertain) {
      clearFaceOverlay();
      liveMessage(activeOperation() || operationUncertain || data?.state === "paused" ? "촬영에 집중하고 있어요. 끝나면 이름이 다시 표시돼요."
        : data?.state === "running" ? "얼굴을 확인하고 있어요."
        : data?.message || "라이브 인식 서비스를 확인하고 있습니다.");
      return;
    }
    const size = data.image_size;
    const sequence = Number(lastSequence);
    // Boxes refer to the same aligned sensor image as the preview. Do not paint
    // a future result over an older displayed frame, or retain old names on a stalled preview.
    const aligned = Array.isArray(size) && size.length === 2 && size.every((n) => finite(n) && n > 0 && n <= 4096)
      && size[0] === image.naturalWidth && size[1] === image.naturalHeight
      && Number.isSafeInteger(data.sequence) && Number.isSafeInteger(sequence) && data.sequence <= sequence
      && displayedCaptureAt !== null && Math.abs(displayedCaptureAt - (facesReceivedAt - facesAgeAtReceipt)) < MAX_FACE_AGE_MS;
    if (!aligned) {
      clearFaceOverlay();
      liveMessage("영상과 얼굴 위치를 맞추고 있습니다.");
      return;
    }
    const [width, height] = size;
    const faces = Array.isArray(data.faces) ? data.faces.slice(0, 16).filter((face) =>
      Array.isArray(face.bbox) && face.bbox.length === 4 && face.bbox.every(finite)
      && face.bbox[2] > face.bbox[0] && face.bbox[3] > face.bbox[1]
      && ["known", "unknown", "uncertain"].includes(face.status)) : [];
    liveMessage(faces.length ? `${faces.length}명을 찾았어요${data.truncated ? " · 일부 얼굴만 표시 중" : ""} · 가장 큰 얼굴로 인증해요.`
      : "카메라를 바라보면 얼굴과 이름이 표시돼요.");
    const key = JSON.stringify([data.sequence, size, faces]);
    if (key === facesRenderKey) return;
    clearFaceOverlay();
    facesRenderKey = key;
    if (!faces.length) return;
    const overlay = $("face-overlay");
    overlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
    overlay.toggleAttribute("hidden", false);
    const svg = (tag, attrs) => {
      const element = document.createElementNS(SVG_NS, tag);
      for (const [name, value] of Object.entries(attrs)) element.setAttribute(name, String(value));
      return element;
    };
    for (const face of faces) {
      const x1 = Math.max(0, Math.min(width, face.bbox[0]));
      const y1 = Math.max(0, Math.min(height, face.bbox[1]));
      const x2 = Math.max(0, Math.min(width, face.bbox[2]));
      const y2 = Math.max(0, Math.min(height, face.bbox[3]));
      if (x2 <= x1 || y2 <= y1) continue;
      // Mirror coordinates once; keep SVG text in its ordinary reading direction.
      const x = width - x2;
      const group = svg("g", { class: "face-mark", "data-status": face.status, "data-primary": face.primary === true });
      const box = svg("rect", { class: "face-box", x, y: y1, width: x2 - x1, height: y2 - y1, rx: 5 });
      const name = face.status === "known" && typeof face.display_name === "string" ? face.display_name : face.status === "unknown" ? "미등록" : "확인 중";
      const characters = Array.from(name);
      const caption = `${characters.slice(0, 24).join("")}${characters.length > 24 ? "…" : ""}${face.primary === true ? " · 인증 우선 대상" : ""}`;
      const title = svg("title", {});
      title.textContent = `${name}${face.primary === true ? " · 인증 우선 대상" : ""}`;
      const background = svg("rect", { class: "face-label-bg", rx: 4, height: 28 });
      const label = svg("text", { class: "face-label", "dominant-baseline": "middle" });
      label.textContent = caption;
      group.append(title, box, background, label);
      overlay.append(group);
      const labelWidth = Math.min(width - 8, Math.ceil(label.getComputedTextLength()) + 16);
      const labelX = Math.max(4, Math.min(x, width - labelWidth - 4));
      const labelY = Math.max(4, y1 >= 32 ? y1 - 31 : Math.min(y1 + 4, height - 32));
      background.setAttribute("x", labelX);
      background.setAttribute("y", labelY);
      background.setAttribute("width", labelWidth);
      label.setAttribute("x", labelX + 8);
      label.setAttribute("y", labelY + 14);
    }
  }

  async function pollFaces() {
    if (faceBusy || document.hidden || !isStreaming()) return;
    faceBusy = true;
    const controller = new AbortController();
    faceController = controller;
    const requestedAt = performance.now();
    const timeout = setTimeout(() => controller.abort(), 1500);
    try {
      const response = await fetch("/api/live/faces", { cache: "no-store", credentials: "same-origin", signal: controller.signal });
      if (!response.ok) throw new Error("Live recognition unavailable");
      const data = await response.json();
      if (document.hidden) return;
      if (!data || typeof data.state !== "string") throw new Error("Invalid live recognition status");
      latestFaces = data;
      facesReceivedAt = performance.now();
      // Include request time conservatively; server and browser clocks need not match.
      facesAgeAtReceipt = finite(data.age_ms) ? Math.max(0, data.age_ms) + facesReceivedAt - requestedAt : Infinity;
      renderFaces();
    } catch {
      latestFaces = null;
      clearFaceOverlay();
      $("recognition-fps").textContent = "—";
      liveMessage("라이브 인식 연결을 기다리고 있습니다.");
    } finally {
      clearTimeout(timeout);
      if (faceController === controller) faceController = null;
      faceBusy = false;
    }
  }

  function renderStatus(status) {
    const camera = status.camera || {};
    const streaming = isStreaming();
    metrics(streaming ? status.capture : { ...status.capture, fps: null, depth_median_mm: null, depth_valid_ratio: null });
    $("detail-server").textContent = "연결됨";
    $("detail-mode").textContent = status.mode === "authentication" ? "인증" : "센서 미리보기";
    $("detail-updated").textContent = new Date(statusAt).toLocaleTimeString("ko-KR", { hour12: false });
    $("camera-message").textContent = camera.message || "센서에서 상태를 확인하고 있습니다.";
    $("device-info").hidden = !camera.info;
    $("device-info").textContent = camera.info
      ? (typeof camera.info === "string" ? camera.info : JSON.stringify(camera.info, null, 2)) : "";

    if (streaming) {
      badge("camera-state", "영상 수신 중", "good");
      if (frameDisplayed) indicator("실시간 영상", "good");
    } else if (camera.state === "error") {
      badge("camera-state", "연결 오류", "error");
      indicator("센서 연결 오류", "error");
      clearPreview("카메라 연결을 확인해 주세요", "센서 상태와 운영자 진단 정보를 확인하세요.");
    } else if (camera.state === "streaming") {
      badge("camera-state", "영상 지연", "warning");
      indicator("새 프레임 대기", "warning");
      clearPreview("새 영상을 기다리고 있어요", "센서에서 최신 프레임이 수신되면 자동으로 다시 표시됩니다.");
    } else {
      badge("camera-state", "연결 대기", "waiting");
      indicator("센서 연결 대기", "waiting");
      clearPreview("카메라를 준비하고 있어요", "센서에서 영상이 도착하면 여기에 표시됩니다.");
    }

    renderAuth();
  }

  function authState() {
    return latestStatus?.auth || {};
  }

  function activeOperation() {
    return currentOperation && activeStates.has(currentOperation.state);
  }

  function consentReady() {
    const consent = authState().consent;
    return typeof consent?.version === "string" && consent.version.length > 0
      && typeof consent.text === "string" && consent.text.length > 0;
  }

  function consoleMode() {
    return authState().console_mode === true;
  }

  function sessionOwner() {
    const auth = authState();
    return consoleMode() ? `console:${auth.console_session_id || "ready"}` : auth.account?.account_id || null;
  }

  function participantReady() {
    const auth = authState();
    return consoleMode() ? auth.console_ready === true && auth.management_allowed === true
      && accountsReady && Boolean(enrollmentTarget) && enrollmentTarget.enabled !== false
      : auth.account?.authenticated === true && auth.account.enabled !== false;
  }

  function canStart() {
    const auth = authState();
    return auth.enabled === true && auth.ready === true && participantReady()
      && consentReady() && $("biometric-consent").checked && isStreaming()
      && (!enrollmentTarget || (isAdmin() && enrollmentTarget.enabled !== false))
      && !authBusy && !activeOperation() && !operationUncertain;
  }

  function canObserve() {
    const auth = authState();
    return auth.enabled === true && auth.capture_ready === true && participantReady()
      && consentReady() && $("biometric-consent").checked && isStreaming()
      && (consoleMode() || !enrollmentTarget) && !authBusy && !activeOperation() && !operationUncertain;
  }

  function feedback(message, tone = "warning") {
    $("auth-feedback").textContent = message;
    $("auth-feedback").hidden = !message;
    $("auth-feedback").dataset.tone = tone;
  }

  function renderAuth() {
    const auth = authState();
    const enabled = auth.enabled === true;
    const console = consoleMode();
    const padDisabled = enabled && auth.pad_enabled === false;
    const signedIn = auth.account?.authenticated === true;
    const sessionReady = console ? auth.console_ready === true : signedIn;
    const accountId = sessionReady ? sessionOwner() : null;
    if (enabled && (!sessionReady || (signedAccountId !== null && signedAccountId !== accountId))) {
      currentOperation = null;
      operationId = null;
      operationUncertain = false;
      cancelRequested = false;
      pendingRequestId = null;
      $("operation-panel").hidden = true;
      $("biometric-consent").checked = false;
      $("photo-consent").checked = false;
      resetAccountManagement();
    }
    if (enabled) signedAccountId = accountId;
    const busy = authBusy || Boolean(activeOperation()) || operationUncertain;
    const blockers = Array.isArray(auth.blockers) ? auth.blockers : [];
    const list = $("auth-blockers");
    list.replaceChildren();
    for (const blocker of blockers) {
      const item = document.createElement("li");
      item.textContent = blocker.message || blocker.code || "인증 구성 요소를 확인해 주세요.";
      list.append(item);
    }
    list.hidden = blockers.length === 0;
    $("auth-workflow").hidden = !enabled;
    document.body.dataset.authEnabled = String(enabled);
    document.body.dataset.consoleMode = String(console);
    document.body.dataset.accountState = sessionReady ? "signed-in" : "guest";
    document.body.dataset.accountMode = accountMode;
    $("pilot-notice").hidden = !enabled || auth.pilot !== true;
    $("pad-disabled-notice").hidden = !padDisabled;
    $("page-title").textContent = enabled ? "얼굴로, 간편하게" : "카메라를 만나보세요";
    $("page-description").textContent = enabled
      ? "등록부터 확인까지, 카메라 앞에서 한 번에."
      : "카메라를 연결하면 실시간으로 내 모습을 볼 수 있어요.";
    document.title = enabled ? "PaceAuth · 얼굴 등록 · 인증" : "PaceAuth · 카메라 테스트";
    $("footer-mode").textContent = enabled ? "얼굴 인식 체험" : "카메라 미리보기";
    document.querySelector(".device-label").lastChild.textContent = console ? "관리 · 데모" : "내부망 체험";
    $("pilot-notice").textContent = console ? "등록한 계정으로 얼굴 일치를 체험하는 데모예요. 실제 서비스의 계정 본인 확인을 대신하지 않습니다."
      : "체험용 테스트 계정을 사용하며, 실제 서비스의 계정 본인 확인을 대신하지 않습니다.";
    $("auth-heading").textContent = console ? "누구나, 얼굴로 간편하게" : "내 얼굴로 시작하기";
    $("journey-account-label").textContent = console ? "계정 선택" : "계정";
    if (auth.ready === true && enabled) {
      badge("auth-state", padDisabled ? "PAD 꺼짐" : "준비 완료", padDisabled ? "warning" : "good");
      $("auth-description").textContent = padDisabled
        ? "얼굴을 등록하고 일치 여부를 확인해 보세요."
        : "얼굴을 등록하고 간편하게 인증해 보세요.";
    } else {
      badge("auth-state", "사용 준비 필요", "warning");
      $("auth-description").textContent = "얼굴 등록·인증을 사용하려면 아래 준비 항목을 완료해야 합니다.";
      if (!blockers.length) $("auth-description").textContent = "인증 서비스가 아직 준비되지 않았습니다. 현재 센서 연결과 영상만 확인할 수 있습니다.";
    }
    $("pilot-login-form").hidden = console || signedIn;
    $("signup-fields").hidden = accountMode !== "signup";
    $("pilot-login").hidden = accountMode !== "login";
    $("pilot-signup").hidden = accountMode !== "signup";
    for (const mode of ["login", "signup"]) {
      const button = $("account-mode-" + mode);
      button.setAttribute("aria-pressed", String(accountMode === mode));
      button.disabled = !enabled || busy;
    }
    $("pilot-password").autocomplete = accountMode === "signup" ? "new-password" : "current-password";
    $("verified-account").hidden = console || !signedIn;
    $("account-name").textContent = signedIn ? (auth.account.display_name || auth.account.account_id || "체험 계정") : "";
    $("profile-form").hidden = console || !signedIn;
    $("profile-settings").hidden = console || !signedIn;
    if (profileAccountId !== accountId) {
      profileAccountId = accountId;
      profileDirty = false;
    }
    if (!profileDirty) $("profile-display-name").value = signedIn ? (auth.account.display_name || auth.account.account_id || "") : "";
    $("profile-display-name").disabled = !signedIn || busy;
    $("profile-save").disabled = !signedIn || busy;
    for (const id of ["pilot-display-name", "pilot-account", "pilot-password", "pilot-login", "pilot-signup", "pilot-logout"]) {
      $(id).disabled = !enabled || busy;
    }
    const revision = consentReady() ? `${auth.consent.version}\n${auth.consent.text}` : null;
    if (revision !== consentRevision) {
      $("biometric-consent").checked = false;
      $("photo-consent").checked = false;
      consentRevision = revision;
    }
    const hasParticipant = console ? Boolean(enrollmentTarget) : signedIn;
    const flowStep = !hasParticipant ? "account" : !consentReady() || !$("biometric-consent").checked
      ? "consent" : busy ? "busy" : "ready";
    document.body.dataset.flowStep = flowStep;
    for (const item of document.querySelectorAll(".journey-step[data-step]")) {
      if (item.dataset.step === (flowStep === "busy" ? "ready" : flowStep)) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
    }
    $("consent-description").textContent = consentReady() ? auth.consent.text
      : "생체정보 처리 안내를 불러오고 있습니다. 안내 확인 전에는 등록·인증을 시작할 수 없습니다.";
    $("consent-version").textContent = consentReady() ? `안내 버전 ${auth.consent.version}` : "";
    $("biometric-consent").disabled = !enabled || !hasParticipant || !consentReady() || busy;
    $("photo-consent").disabled = !enabled || !hasParticipant || !consentReady() || busy;
    renderConsolePicker(busy);
    $("enrollment-target").hidden = !enrollmentTarget;
    $("enrollment-target-label").textContent = console ? "선택한 계정" : "지금 얼굴을 등록할 사람";
    $("enrollment-target-name").textContent = enrollmentTarget?.display_name || enrollmentTarget?.account_id || "";
    $("enrollment-target-id").textContent = enrollmentTarget?.account_id || "";
    $("enrollment-target-note").textContent = console && enrollmentTarget?.face_status === "active"
      ? "바로 인증할 수 있어요. 다시 등록하면 기존 얼굴과 사진을 새 촬영으로 바꿔요. 사진은 동의한 경우에만 저장해요."
      : enrollmentTarget && needsReEnrollment(enrollmentTarget)
      ? "기존 얼굴과 사진을 새 촬영으로 바꿔요. 사진 저장에 동의하지 않으면 새 사진은 저장하지 않아요. 본인이 안내를 읽고 동의해 주세요."
      : "등록할 본인이 안내를 읽고 동의한 뒤, 카메라 앞에 혼자 서 주세요.";
    $("clear-enrollment-target").disabled = busy;
    $("start-enroll").textContent = enrollmentTarget && needsReEnrollment(enrollmentTarget) ? "얼굴 다시 등록" : "얼굴 등록";
    $("start-enroll").disabled = !canStart();
    $("start-authenticate").hidden = !console && Boolean(enrollmentTarget);
    $("start-authenticate").disabled = !canStart() || (console && enrollmentTarget?.face_status !== "active");
    $("retry-operation").disabled = !(lastOperationKind === "observe" ? canObserve() : canStart());
    $("observation-panel").hidden = !enabled;
    for (const control of $("observation-form").elements) control.disabled = !enabled || busy;
    const attack = $("observation-kind").value !== "live";
    $("observation-instrument").disabled = !enabled || busy || !attack;
    $("observation-instrument").required = attack;
    $("start-observation").disabled = !canObserve();
    const observationMode = $("observation-mode").value;
    $("observation-mode-description").textContent = (observationModes[observationMode] || observationModes.diagnostic).description;
    $("observation-note").textContent = !auth.capture_ready ? "센서와 분석 서비스의 준비가 완료되면 표본을 수집할 수 있습니다."
      : `${console ? "계정 선택" : "계정 로그인"}과 생체정보 처리 동의가 필요합니다. ${observationMode === "enroll" ? "카메라에서 50~80cm 거리를 유지하고 9방향 촬영 안내에 따라 천천히 움직여 주세요." : "정면을 바라보고 카메라에서 50~80cm 거리를 유지해 주세요."}`;
    $("cancel-operation").disabled = authBusy || cancelRequested || currentOperation?.state === "processing";
    $("cancel-operation").textContent = cancelRequested ? "취소 확인 중…"
      : currentOperation?.state === "processing" ? "서버 처리 중" : "촬영 취소";
    $("action-note").textContent = !enabled ? "현재 화면은 센서 확인용이며 얼굴 등록·인증은 진행하지 않습니다."
      : operationUncertain ? "요청을 확인하고 있어요. 잠시만 기다려 주세요."
      : activeOperation() ? "카메라 안내를 따라 주세요. 결과가 나오면 알려드릴게요."
      : !hasParticipant ? console ? "계정을 선택하거나 새 계정을 추가해 주세요." : "내 계정으로 로그인하면 시작할 수 있어요."
      : !auth.ready ? "위 준비 항목이 모두 해결되면 등록·인증을 사용할 수 있습니다."
      : !$("biometric-consent").checked ? "안내를 읽고 동의하면 준비 끝이에요."
      : "등록은 혼자, 인증은 가장 크게 보이는 얼굴로 진행해요.";
    renderAccountManagement();
    renderEnrollment();
    if (console && isAdmin() && !accountsLoading && Date.now() - accountsFetchedAt > 5000 && !busy) void loadAccounts();
    if (sessionReady && auth.operation_id && auth.operation_id !== dismissedOperationId && !operationId && !pendingRequestId) {
      operationId = String(auth.operation_id);
      operationUncertain = true;
      renderAuth();
      void pollOperation();
    }
  }

  function isAdmin() {
    if (consoleMode()) return authState().console_ready === true && authState().management_allowed === true;
    const account = authState().account;
    return account?.authenticated === true && account.is_admin === true && account.enabled !== false;
  }

  function canManage() {
    return isAdmin() && !authBusy && !activeOperation() && !operationUncertain;
  }

  function needsReEnrollment(account) {
    return Boolean(account && ["active", "pending_review", "disabled"].includes(account.face_status));
  }

  function managedAccount() {
    return managedAccounts.find((account) => account.account_id === selectedAccountId) || null;
  }

  function selectParticipant(account) {
    enrollmentTarget = account ? { ...account } : null;
    $("biometric-consent").checked = false;
    $("photo-consent").checked = false;
    dismissedOperationId = operationId || dismissedOperationId;
    currentOperation = null;
    operationId = null;
    pendingRequestId = null;
    $("operation-panel").hidden = true;
    feedback("");
  }

  function renderConsolePicker(busy) {
    const enabled = consoleMode();
    $("console-account-picker").hidden = !enabled;
    if (!enabled) return;
    const choices = managedAccounts.filter((account) => account.enabled !== false);
    const key = JSON.stringify([choices.map((account) => [account.account_id, account.display_name, account.face_status]), enrollmentTarget?.account_id]);
    const select = $("demo-account");
    if (key !== consolePickerKey) {
      consolePickerKey = key;
      const placeholder = new Option("계정을 선택해 주세요", "");
      select.replaceChildren(placeholder);
      for (const account of choices) {
        const name = account.display_name || account.account_id;
        const duplicateName = choices.some((other) => other.account_id !== account.account_id && (other.display_name || other.account_id) === name);
        select.append(new Option(`${name}${duplicateName ? ` · ${account.account_id}` : ""} · ${faceState(account)[0]}`, account.account_id));
      }
      select.value = enrollmentTarget?.account_id || "";
    }
    select.disabled = busy || !accountsReady || !choices.length;
    $("console-add-account").disabled = !canManage();
    $("demo-account-note").textContent = !accountsLoaded ? "계정 목록을 불러오고 있어요."
      : !accountsReady ? "계정 서비스 연결을 확인하고 있어요."
      : !choices.length ? "사용할 계정을 추가하거나, 계정 관리에서 활성화해 주세요."
      : !enrollmentTarget ? "등록하거나 인증할 사람을 선택해 주세요."
      : enrollmentTarget.face_status === "active" ? "등록된 얼굴로 일치 여부를 확인할 수 있어요."
      : "얼굴을 먼저 등록하면 인증을 체험할 수 있어요.";
  }

  function accountFeedback(message, tone = "warning", id = "accounts-feedback") {
    $(id).textContent = message;
    $(id).hidden = !message;
    $(id).dataset.tone = tone;
  }

  function resetAccountManagement() {
    mainView = "live";
    managedAccounts = [];
    accountsLoaded = false;
    accountsReady = false;
    accountsOwner = null;
    accountsFetchedAt = 0;
    consolePickerKey = null;
    selectedAccountId = null;
    managedProfileDirty = false;
    accountsRenderKey = null;
    detailRenderKey = null;
    enrollmentTarget = null;
    deletionTarget = null;
    $("account-list").replaceChildren();
    $("account-detail-avatar").replaceChildren();
    $("create-account-dialog").close();
    $("delete-account-dialog").close();
    $("new-account-password").value = "";
  }

  function setMainView(view) {
    if (view === "accounts" && !canManage()) return;
    mainView = view === "accounts" ? "accounts" : "live";
    renderAccountManagement();
    renderEnrollment();
    if (mainView === "accounts") {
      $("accounts-heading").focus({ preventScroll: true });
      void loadAccounts();
    }
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  function dateLabel(value) {
    if (!value) return "";
    const date = new Date(typeof value === "number" && value < 100000000000 ? value * 1000 : value);
    return Number.isFinite(date.getTime()) ? date.toLocaleDateString("ko-KR") : "";
  }

  function faceState(account) {
    const states = {
      active: ["얼굴 등록됨", "good"], none: ["얼굴 미등록", "waiting"],
      pending_review: ["등록 확인 필요", "warning"], disabled: ["얼굴 사용 중지", "waiting"],
      revoked: ["얼굴 등록 해제", "waiting"], unknown: ["얼굴 상태 확인 중", "warning"],
    };
    return states[account.face_status] || states.unknown;
  }

  function personAvatar(container, account) {
    container.replaceChildren();
    container.dataset.disabled = String(account.enabled === false);
    const initial = document.createElement("span");
    initial.textContent = Array.from(account.display_name || account.account_id || "?")[0];
    container.append(initial);
    if (!account.has_photo) return;
    const photo = document.createElement("img");
    photo.alt = "";
    photo.loading = "lazy";
    photo.decoding = "async";
    photo.src = `/api/auth/accounts/${encodeURIComponent(account.account_id)}/photo?v=${encodeURIComponent(account.photo_version || account.photo_updated_at || "0")}`;
    photo.addEventListener("error", () => photo.remove(), { once: true });
    container.append(photo);
  }

  function renderAccountList() {
    const query = $("account-search").value.trim().toLocaleLowerCase();
    const key = JSON.stringify([managedAccounts, query, selectedAccountId]);
    if (key === accountsRenderKey) return;
    accountsRenderKey = key;
    const visible = managedAccounts.filter((account) => `${account.display_name || ""} ${account.account_id}`.toLocaleLowerCase().includes(query));
    const focusedAccount = document.activeElement?.dataset.accountId;
    const list = $("account-list");
    list.replaceChildren();
    $("accounts-count").textContent = managedAccounts.length.toLocaleString("ko-KR");
    for (const account of visible) {
      const item = document.createElement("div");
      item.setAttribute("role", "listitem");
      const card = document.createElement("button");
      card.type = "button";
      card.className = "person-card";
      card.dataset.accountId = account.account_id;
      card.setAttribute("aria-pressed", String(account.account_id === selectedAccountId));
      const top = document.createElement("span");
      top.className = "person-card-top";
      const avatar = document.createElement("span");
      avatar.className = "person-avatar";
      avatar.setAttribute("aria-hidden", "true");
      personAvatar(avatar, account);
      top.append(avatar);
      if (account.is_admin && !consoleMode()) {
        const admin = document.createElement("span");
        admin.className = "person-admin";
        admin.textContent = "관리자";
        top.append(admin);
      }
      const name = document.createElement("strong");
      name.className = "person-name";
      name.textContent = account.display_name || account.account_id;
      const id = document.createElement("span");
      id.className = "person-id";
      id.textContent = account.account_id;
      const statuses = document.createElement("span");
      statuses.className = "person-status";
      for (const [label, tone] of [faceState(account), ...(account.enabled === false ? [["비활성", "waiting"]] : [])]) {
        const pill = document.createElement("span");
        pill.className = "status-pill";
        pill.dataset.tone = tone;
        pill.textContent = label;
        statuses.append(pill);
      }
      card.append(top, name, id, statuses);
      card.addEventListener("click", () => {
        if (!canManage()) return;
        selectedAccountId = account.account_id;
        managedProfileDirty = false;
        renderAccountManagement();
        if (window.matchMedia("(max-width: 840px)").matches) $("account-detail").scrollIntoView({
          block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
        });
      });
      item.append(card);
      list.append(item);
      if (focusedAccount === account.account_id) card.focus({ preventScroll: true });
    }
    $("accounts-empty").hidden = visible.length > 0;
    $("accounts-empty-title").textContent = accountsLoading && !accountsLoaded ? "계정을 불러오고 있어요"
      : !accountsLoaded ? "계정 목록을 확인해 주세요" : query ? "일치하는 계정이 없어요" : "첫 계정을 추가해 보세요";
    $("accounts-empty-description").textContent = accountsLoading && !accountsLoaded ? "잠시만 기다려 주세요."
      : !accountsLoaded ? "새로고침을 눌러 다시 불러올 수 있어요." : query ? "다른 이름이나 아이디로 검색해 주세요." : "계정을 만들고 얼굴을 등록하면 준비 끝이에요.";
  }

  function renderAccountManagement() {
    const admin = isAdmin();
    const busy = !canManage();
    $("main-navigation").hidden = !admin;
    if (!admin || activeOperation() || operationUncertain) mainView = "live";
    const managing = admin && mainView === "accounts";
    document.body.dataset.mainView = managing ? "accounts" : "live";
    $("accounts-page").hidden = !managing;
    $("live-intro").hidden = managing;
    $("live-workspace").hidden = managing;
    $("live-advanced").hidden = managing;
    $("nav-accounts").disabled = busy;
    for (const [id, active] of [["nav-live", !managing], ["nav-accounts", managing]]) {
      if (active) $(id).setAttribute("aria-current", "page");
      else $(id).removeAttribute("aria-current");
    }
    if (!admin) return;
    const console = consoleMode();
    $("new-account").disabled = busy;
    $("refresh-accounts").disabled = busy || accountsLoading;
    $("accounts-service-message").hidden = !accountsLoaded || accountsReady;
    $("accounts-service-message").textContent = accountsMessage || "얼굴 등록 서비스 연결을 확인하고 있어요. 이름과 계정 정보를 확인할 수 있어요.";
    for (const control of $("create-account-form").elements) control.disabled = busy;
    $("new-account-password-field").hidden = console;
    $("new-account-password").disabled = console || busy;
    $("new-account-password").required = !console;
    $("new-account-id").required = !console;
    $("new-account-id").placeholder = console ? "비워 두면 자동으로 만들어요" : "영문 소문자, 숫자 등 3~64자";
    $("new-account-id-optional").hidden = !console;
    $("cancel-delete-account").disabled = busy;
    $("confirm-delete-account").disabled = busy;
    if (!managing) return;
    renderAccountList();
    for (const card of $("account-list").querySelectorAll("button")) card.disabled = busy;
    const account = managedAccount();
    $("account-selection-empty").hidden = Boolean(account);
    $("account-detail").hidden = !account;
    if (!account) return;
    const key = JSON.stringify(account);
    if (key !== detailRenderKey) {
      detailRenderKey = key;
      personAvatar($("account-detail-avatar"), account);
    }
    $("account-detail-name").textContent = account.display_name || account.account_id;
    $("account-detail-id").textContent = account.account_id;
    $("account-detail-admin").hidden = console || !account.is_admin;
    const [faceLabel, faceTone] = faceState(account);
    badge("account-detail-face", faceLabel, faceTone);
    badge("account-detail-enabled", account.enabled === false ? "비활성" : "활성", account.enabled === false ? "waiting" : "good");
    if (!managedProfileDirty) $("managed-display-name").value = account.display_name || account.account_id;
    $("managed-display-name").disabled = busy;
    $("save-managed-profile").disabled = busy;
    $("delete-account-photo").hidden = !account.has_photo;
    $("delete-account-photo").disabled = busy;
    $("account-photo-description").textContent = account.has_photo
      ? "동의하고 촬영한 등록 사진이에요. 사진만 삭제해도 얼굴 등록은 유지돼요."
      : needsReEnrollment(account) ? "저장된 사진이 없어요. 사진을 남기려면 동의 후 얼굴을 다시 촬영해 주세요."
      : "얼굴을 등록할 때, 동의한 경우에만 정면 사진 1장을 저장해요.";
    $("enroll-managed-account").textContent = needsReEnrollment(account) ? "사진 · 얼굴 다시 등록" : "얼굴 등록하기";
    $("enroll-managed-account").disabled = busy || !accountsReady || account.enabled === false || account.face_status === "unknown";
    const enrolled = dateLabel(account.enrolled_at);
    $("account-enrolled-at").hidden = !enrolled;
    $("account-enrolled-at").textContent = enrolled ? `${enrolled} 얼굴 등록` : "";
    const self = !console && account.account_id === authState().account?.account_id;
    const lastAdmin = !console && account.is_admin && account.enabled !== false && managedAccounts.filter((item) => item.is_admin && item.enabled !== false).length <= 1;
    $("toggle-account-enabled").setAttribute("aria-checked", String(account.enabled !== false));
    $("toggle-account-enabled").setAttribute("aria-label", `${account.display_name || account.account_id} 계정 활성화`);
    $("toggle-account-enabled").disabled = busy || !accountsReady || self || lastAdmin;
    $("account-enabled-description").textContent = account.enabled === false ? "얼굴 인식과 인증에서 제외돼요." : "얼굴 인식과 인증에 사용할 수 있어요.";
    $("account-protection-note").hidden = !self && !lastAdmin;
    $("account-protection-note").textContent = self ? "로그인 중인 내 계정은 비활성화하거나 삭제할 수 없어요." : "마지막 활성 관리자 계정은 유지해야 해요.";
    const created = dateLabel(account.created_at);
    $("account-created-at").textContent = created ? `${created} 계정 생성` : "";
    $("delete-managed-account").disabled = busy || !accountsReady || self || lastAdmin;
  }

  async function loadAccounts() {
    if (!isAdmin() || accountsLoading) return;
    const owner = sessionOwner();
    accountsOwner = owner;
    accountsFetchedAt = Date.now();
    accountsLoading = true;
    accountsRenderKey = null;
    renderAccountManagement();
    try {
      const data = await authRequest("/api/auth/accounts");
      if (!isAdmin() || sessionOwner() !== owner || accountsOwner !== owner) return;
      if (!Array.isArray(data?.accounts)) throw new Error("계정 목록을 확인할 수 없습니다.");
      managedAccounts = data.accounts.filter((account) => typeof account.account_id === "string");
      accountsLoaded = true;
      accountsReady = data.management_ready === true;
      accountsMessage = typeof data.message === "string" ? data.message : "";
      if (selectedAccountId && !managedAccount()) {
        selectedAccountId = null;
        managedProfileDirty = false;
      }
      if (enrollmentTarget) {
        const updated = managedAccounts.find((account) => account.account_id === enrollmentTarget.account_id);
        if (updated && updated.enabled !== false) enrollmentTarget = updated;
        else if (consoleMode()) selectParticipant(null);
      }
    } catch (error) {
      if (accountsOwner !== owner) return;
      accountsReady = false;
      accountFeedback(error.status ? error.message : "계정 목록을 불러오지 못했어요. 연결 상태를 확인하고 다시 시도해 주세요.");
      if (error.status === 401 || error.status === 403) void pollStatus();
    } finally {
      accountsLoading = false;
      accountsRenderKey = null;
      renderAuth();
    }
  }

  async function createManagedAccount() {
    if (!canManage() || !$("create-account-form").reportValidity()) return;
    const accountId = $("new-account-id").value.trim();
    const displayName = $("new-display-name").value.trim();
    const password = $("new-account-password").value;
    if ((!consoleMode() || accountId) && !/^[a-z0-9][a-z0-9_.-]{2,63}$/.test(accountId) || !displayName) {
      accountFeedback("이름과 아이디를 확인해 주세요.", "warning", "create-account-feedback");
      return;
    }
    authBusy = true;
    renderAuth();
    accountFeedback("계정을 만들고 있어요.", "waiting", "create-account-feedback");
    try {
      const data = await authRequest("/api/auth/accounts", { display_name: displayName,
        ...(accountId ? { account_id: accountId } : {}), ...(!consoleMode() ? { password } : {}) });
      $("new-account-password").value = "";
      $("create-account-dialog").close();
      selectedAccountId = data?.account?.account_id || accountId;
      managedProfileDirty = false;
      await loadAccounts();
      accountFeedback(`${displayName}님의 계정을 만들었어요. 본인이 동의하면 얼굴을 등록할 수 있어요.`, "good");
    } catch (error) {
      $("new-account-password").value = "";
      accountFeedback(error.status === 409 ? "이미 사용 중인 아이디예요. 다른 아이디를 입력해 주세요."
        : error.status ? error.message : "생성 결과를 확인하지 못했어요. 창을 닫고 목록을 새로고침해 주세요.", "warning", "create-account-feedback");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function updateManagedAccount(action, body) {
    const account = managedAccount();
    if (!account || !canManage()) return;
    const targetId = account.account_id;
    authBusy = true;
    renderAuth();
    accountFeedback("변경 내용을 저장하고 있어요.", "waiting");
    try {
      await authRequest(`/api/auth/accounts/${encodeURIComponent(targetId)}/${action}`, body, { timeoutMs: action === "status" ? 30000 : 10000 });
      managedProfileDirty = false;
      await loadAccounts();
      if (targetId === authState().account?.account_id) await pollStatus();
      accountFeedback(action === "status" ? body.enabled ? "계정을 활성화했어요." : "계정을 비활성화했어요. 얼굴 인식과 인증에서 제외돼요."
        : "이름을 저장했어요.", "good");
    } catch (error) {
      accountFeedback(error.status ? error.message : "변경 결과를 확인하지 못했어요. 목록을 새로고침해 주세요.");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  function prepareManagedEnrollment() {
    const account = managedAccount();
    if (!account || !canManage() || !accountsReady || account.enabled === false || account.face_status === "unknown") return;
    selectParticipant(account);
    setMainView("live");
    renderAuth();
    $("auth-heading").focus({ preventScroll: true });
    $("enrollment-target").scrollIntoView({ block: "nearest", behavior: "instant" });
  }

  function openDeleteDialog(kind) {
    const account = managedAccount();
    if (!account || !canManage()) return;
    if (kind === "account" && $("delete-managed-account").disabled) return;
    if (kind === "photo" && !account.has_photo) return;
    deletionTarget = { ...account };
    deletionKind = kind;
    $("delete-target-name").textContent = account.display_name || account.account_id;
    $("delete-target-id").textContent = account.account_id;
    $("delete-account-heading").textContent = kind === "photo" ? "등록 사진을 삭제할까요?" : "이 계정을 삭제할까요?";
    $("delete-account-dialog").querySelector(".status-message").textContent = kind === "photo"
      ? "저장된 사진 1장을 삭제해요. 계정과 얼굴 등록 정보는 유지돼요."
      : "계정과 등록 사진, 얼굴 등록 정보를 삭제해요. 삭제 후에는 다시 등록해야 해요.";
    $("confirm-delete-account").textContent = kind === "photo" ? "사진 삭제" : "삭제하기";
    accountFeedback("", "warning", "delete-account-feedback");
    $("delete-account-dialog").showModal();
    $("cancel-delete-account").focus();
  }

  async function deleteManagedAccount() {
    if (!deletionTarget || !canManage()) return;
    const target = deletionTarget;
    const kind = deletionKind;
    authBusy = true;
    renderAuth();
    try {
      await authRequest(`/api/auth/accounts/${encodeURIComponent(target.account_id)}/${kind === "photo" ? "photo/delete" : "delete"}`,
        kind === "photo" ? {} : { confirm_account_id: target.account_id }, { timeoutMs: 60000 });
      $("delete-account-dialog").close();
      deletionTarget = null;
      if (kind === "account") {
        selectedAccountId = null;
        if (enrollmentTarget?.account_id === target.account_id) enrollmentTarget = null;
      }
      await loadAccounts();
      accountFeedback(kind === "photo" ? "등록 사진을 삭제했어요. 얼굴 등록은 유지돼요." : `${target.display_name || target.account_id}님의 계정을 삭제했어요.`, "good");
    } catch (error) {
      accountFeedback(error.status ? error.message : "삭제 결과를 확인하지 못했어요. 목록을 새로고침해 주세요.", "warning", "delete-account-feedback");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  function buildEnrollmentRing() {
    const ticks = $("enrollment-ticks");
    if (ticks.childElementCount) return;
    // Screen directions already match the mirrored preview. Do not mirror this SVG.
    for (const [sector, direction] of enrollmentDirections.entries()) {
      for (let index = 0; index < 9; index += 1) {
        const angle = (-90 + sector * 45 + (index - 4) * 5) * Math.PI / 180;
        const tick = document.createElementNS(SVG_NS, "line");
        tick.classList.add("enrollment-tick");
        tick.dataset.direction = direction;
        tick.dataset.index = String(index);
        tick.dataset.state = "waiting";
        for (const [attribute, value] of Object.entries({
          x1: 220 + 195 * Math.cos(angle), y1: 220 + 195 * Math.sin(angle),
          x2: 220 + 210 * Math.cos(angle), y2: 220 + 210 * Math.sin(angle),
        })) tick.setAttribute(attribute, value.toFixed(3));
        ticks.append(tick);
      }
    }
  }

  function enrollmentFeedback(operation, now = performance.now()) {
    const guide = operation.guide || {};
    const age = guideAgeAtReceipt + now - guideReceivedAt;
    if (operationUncertain) return { state: "connection", label: "연결을 확인하고 있어요", prompt: "작업 상태를 확인하고 있어요. 잠시 기다려 주세요." };
    if (cancelRequested) return { state: "waiting", label: "촬영 취소를 확인하고 있어요" };
    if (guide.done === true) return { state: "complete", label: "촬영을 마쳤어요", prompt: "촬영한 얼굴을 확인하고 있어요. 결과를 잠시 기다려 주세요." };
    if (document.hidden || !isStreaming() || !frameDisplayed) return { state: "camera", label: "카메라 영상을 기다리고 있어요", prompt: "카메라 영상을 기다리고 있어요. 연결 상태를 확인해 주세요." };
    if (!Number.isInteger(guide.sample_seq) || !finite(age)) return { state: "waiting", label: "얼굴 방향을 확인하고 있어요", prompt: "정면을 바라보고 잠시 기다려 주세요." };
    if (age >= MAX_GUIDE_AGE_MS) return { state: "stale", label: "얼굴 방향을 다시 확인하고 있어요", prompt: "움직임을 잠시 멈춰 주세요. 얼굴 방향을 다시 확인하고 있어요." };
    if (guide.quality_ok !== true) {
      const labels = { no_face: "얼굴을 찾고 있어요", multi_face: "한 분씩 촬영해 주세요", small_face: "조금 가까이 와 주세요", blur: "움직임을 잠시 멈춰 주세요", exposure: "얼굴 밝기를 확인해 주세요", depth_invalid: "얼굴 거리를 확인하고 있어요", ir_invalid: "얼굴 영상을 확인하고 있어요" };
      return { state: "blocked", label: labels[guide.reason_code] || "촬영 상태를 확인하고 있어요", prompt: guide.prompt || "정면을 바라보고 잠시 기다려 주세요." };
    }
    if (guide.motion?.valid !== true || ![guide.motion?.yaw, guide.motion?.pitch, guide.motion?.azimuth, guide.motion?.radius].every(finite)) {
      return { state: "pose", label: "얼굴 방향을 다시 맞춰 주세요", prompt: guide.prompt || "정면에서 고개를 조금씩 움직여 주세요." };
    }
    return { state: "tracking", label: "고개 움직임을 따라가고 있어요", prompt: guide.prompt };
  }

  function renderEnrollment() {
    const operation = currentOperation;
    const panel = $("operation-panel");
    const enabled = mainView === "live" && operation?.kind === "enroll" && !panel.hidden;
    document.body.dataset.enrollmentMode = String(Boolean(enabled));
    if (enabled !== enrollmentMode) {
      enrollmentMode = enabled;
      enrollmentRingKey = null;
      enrollmentPoseSequence = null;
      if (enabled) {
        $("enrollment-operation-slot").append(panel, $("auth-feedback"));
        // The same preview request loop supplies the circular RGB view.
        if (selectedView !== "rgb") document.querySelector('[data-view="rgb"]').click();
      } else {
        $("operation-home").after(panel);
        $("auth-feedback-home").after($("auth-feedback"));
        schedulePreview();
      }
    }
    for (const id of ["enrollment-header", "enrollment-camera-disc", "enrollment-ring", "enrollment-operation-slot", "enrollment-mode-note"]) {
      // Use the attribute for both HTML and SVG; SVG .hidden is not reflected.
      $(id).toggleAttribute("hidden", !enabled);
    }
    if (!enabled) {
      $("enrollment-guidance").hidden = true;
      $("enrollment-connection-message").hidden = true;
      document.body.dataset.enrollmentTracking = "false";
      document.body.dataset.enrollmentFeedback = "idle";
      $("enrollment-pose-marker").toggleAttribute("hidden", true);
      $("enrollment-finish").hidden = true;
      return;
    }
    buildEnrollmentRing();
    const guide = operation.guide || {};
    const coverage = guide.coverage && typeof guide.coverage === "object" ? guide.coverage : {};
    const hint = Object.hasOwn(guide, "hint_direction") ? guide.hint_direction : guide.current;
    const direction = Object.hasOwn(enrollmentDirectionNames, hint) ? hint : null;
    const feedback = enrollmentFeedback(operation);
    const tracking = operation.state === "capturing" && feedback.state === "tracking";
    const finished = !activeStates.has(operation.state);
    const success = operation.state === "succeeded" && operation.result?.status === "active";
    document.body.dataset.enrollmentTracking = String(tracking);
    document.body.dataset.enrollmentPhase = operation.state;
    document.body.dataset.enrollmentFeedback = feedback.state;
    $("enrollment-guidance").dataset.direction = direction || "sweep";
    $("enrollment-guidance").hidden = !activeStates.has(operation.state);
    $("enrollment-finish").hidden = !finished;
    $("enrollment-finish").dataset.success = String(success);
    $("enrollment-finish-label").textContent = success ? "등록 완료" : operation.state === "cancelled" ? "촬영 취소" : "다시 확인해 주세요";
    $("enrollment-pose-marker").toggleAttribute("hidden", !tracking);
    const poseSequence = `${operation.id}:${guide.sample_seq}`;
    if (tracking && poseSequence !== enrollmentPoseSequence) {
      enrollmentPoseSequence = poseSequence;
      const nx = -guide.motion.yaw / 25;
      const ny = -guide.motion.pitch / 15;
      const radius = Math.hypot(nx, ny);
      const scale = 164 / Math.max(1, radius);
      // Interpolate only between measured positions; never predict or fill capture progress.
      $("enrollment-pose-marker").style.transform = `translate(${(nx * scale).toFixed(2)}px, ${(ny * scale).toFixed(2)}px)`;
      const head = document.querySelector(".enrollment-motion");
      head.style.setProperty("--turn-y", `${-guide.motion.yaw}deg`);
      head.style.setProperty("--turn-x", `${guide.motion.pitch}deg`);
      head.style.setProperty("--feature-x", `${-guide.motion.yaw / 9}px`);
      head.style.setProperty("--feature-y", `${-guide.motion.pitch / 7}px`);
    }
    if (!tracking) enrollmentPoseSequence = null;
    const values = {};
    let capturedFraction = 0;
    let completeDirections = 0;
    for (const key of ["C", ...enrollmentDirections]) {
      const pair = coverage[key];
      const need = Array.isArray(pair) && finite(pair[1]) && pair[1] > 0 ? pair[1] : 0;
      const have = need && finite(pair[0]) ? Math.max(0, Math.min(pair[0], need)) : 0;
      values[key] = { have, need, fraction: need ? have / need : 0, done: need > 0 && have >= need };
      capturedFraction += values[key].fraction;
      if (values[key].done) completeDirections += 1;
    }
    // Only accepted coverage fills ticks. A target or ring_frac never counts as captured data.
    const ringKey = JSON.stringify([values, direction, tracking]);
    if (ringKey !== enrollmentRingKey) {
      enrollmentRingKey = ringKey;
      for (const tick of $("enrollment-ticks").children) {
        const value = values[tick.dataset.direction];
        tick.dataset.state = Number(tick.dataset.index) < Math.floor(value.fraction * 9) ? "done"
          : tracking && tick.dataset.direction === direction ? "current" : "waiting";
      }
      const center = $("enrollment-center-status");
      center.dataset.state = values.C.done ? "done" : tracking && direction === "C" ? "current" : "waiting";
      center.querySelector(".center-status-icon").textContent = values.C.done ? "✓" : "◎";
      $("enrollment-center-label").textContent = values.C.done ? "정면 완료" : "정면";
      const marker = $("enrollment-target-marker");
      const hideTarget = !direction || direction === "C" || !tracking || values[direction].done;
      marker.hidden = hideTarget;
      // SVG elements need the attribute as well; .hidden is not reflected in every browser.
      marker.toggleAttribute("hidden", hideTarget);
      if (direction && direction !== "C") marker.setAttribute("transform", `rotate(${enrollmentDirections.indexOf(direction) * 45} 220 220)`);
    }
    // Missing direction data remains incomplete, even if every reported bin is full.
    const percentage = Math.floor(capturedFraction * 100 / 9);
    $("enrollment-ring").setAttribute("aria-valuenow", String(percentage));
    $("enrollment-ring").setAttribute("aria-valuetext", `${completeDirections} / 9개 방향 완료, 촬영 ${percentage}%`);
    $("enrollment-coverage-count").textContent = `${completeDirections} / 9개 방향 완료 · ${percentage}%`;
    $("enrollment-next-direction").textContent = operation.state === "processing" ? "촬영한 얼굴을 확인하고 있어요"
      : !tracking ? feedback.label
      : !direction ? "파란 점을 보며 천천히 움직여 주세요"
      : direction === "C" ? "먼저 정면을 바라봐 주세요" : `다음 촬영 방향 · ${enrollmentDirectionNames[direction]}`;
    $("enrollment-heading").textContent = operation.state === "processing" ? "내 얼굴을 확인하고 있어요"
      : operation.state === "succeeded" && operation.result?.status === "active" ? "내 얼굴, 등록 완료!"
      : operation.state === "cancelled" ? "촬영을 취소했어요"
      : !activeStates.has(operation.state) ? "얼굴 등록을 다시 확인해 주세요"
      : operation.state === "capturing" && !tracking ? feedback.label
      : values.C.done ? "고개를 천천히 움직여 주세요" : "얼굴을 원 안에 맞춰 주세요";
    $("enrollment-subheading").textContent = activeStates.has(operation.state)
      ? operation.state === "processing" ? "촬영은 끝났어요. 결과를 잠시 기다려 주세요." : tracking ? "파란 점은 지금 내 고개 방향이에요." : "확인된 촬영 진행은 그대로 유지돼요."
      : operation.state === "succeeded" && operation.result?.status === "active" ? "다음부터는 얼굴로 간편하게 확인해요." : "아래 안내와 결과를 확인해 주세요.";
    const connectionMessage = operation.state === "capturing" && ["connection", "camera", "stale"].includes(feedback.state) ? feedback.prompt : "";
    if ($("enrollment-connection-message").textContent !== connectionMessage) $("enrollment-connection-message").textContent = connectionMessage;
    $("enrollment-connection-message").hidden = !connectionMessage;
    if (operation.state === "capturing" && feedback.prompt && feedback.state !== "tracking") {
      if ($("operation-prompt").textContent !== feedback.prompt) $("operation-prompt").textContent = feedback.prompt;
      $("operation-description").textContent = guide.done === true ? "결과가 나오면 알려드릴게요." : "확인되면 촬영을 자동으로 이어가요.";
    }
    $("enrollment-mode-note").textContent = `${(operation.pad_enabled ?? authState().pad_enabled) === false ? "위조 방지 검사 꺼짐 · " : ""}등록 사진은 동의한 경우에만 저장해요.\n불편하면 언제든 촬영을 취소할 수 있어요.`;
  }

  function resultLabel(operation) {
    const result = operation.result || {};
    if (operation.state === "cancelled") return ["촬영 취소", "촬영이 취소되었습니다.", "waiting"];
    if (operation.kind === "observe" && result.recorded === true && operation.state === "succeeded") {
      return ["표본 기록 완료", "검증용 표본을 기록했습니다.", "good"];
    }
    if (operation.kind === "enroll" && result.status === "active" && operation.state === "succeeded") {
      return ["등록 완료", "얼굴 등록이 완료되었습니다.", "good"];
    }
    if (result.status === "pending_review" || result.decision === "step_up") {
      return ["추가 확인 필요", result.status === "pending_review" ? "등록 정보를 운영자가 확인해야 합니다." : "추가 인증이 필요합니다.", "warning"];
    }
    if (operation.kind === "authenticate" && result.decision === "approved" && operation.state === "succeeded") {
      const resultPadEnabled = result.pad_enabled ?? operation.pad_enabled ?? authState().pad_enabled;
      if (resultPadEnabled === false) {
        return ["얼굴 일치", "얼굴이 일치합니다. 위조 방지 검사는 꺼져 있습니다.", "good"];
      }
      return ["인증 성공", "얼굴 인증이 완료되었습니다.", "good"];
    }
    if (operation.state === "rejected") return ["인증되지 않음", "요청이 승인되지 않았습니다.", "warning"];
    return ["완료하지 못함", "요청을 완료하지 못했습니다.", "error"];
  }

  function renderOperation(operation, requestElapsed = 0) {
    if (!operation || typeof operation.id !== "string") return;
    if (pendingRequestId && operation.client_request_id !== pendingRequestId) {
      operationUncertain = true;
      feedback("새 요청의 상태를 확인하고 있습니다. 이전 작업 결과는 새 결과로 표시하지 않습니다.");
      renderAuth();
      return;
    }
    if (operationUncertain) feedback("");
    currentOperation = operation;
    guideReceivedAt = performance.now();
    guideAgeAtReceipt = finite(operation.guide_age_ms) && operation.guide_age_ms >= 0
      ? operation.guide_age_ms + Math.max(0, requestElapsed) : Infinity;
    operationId = operation.id;
    lastOperationKind = operation.kind;
    operationUncertain = false;
    pendingRequestId = null;
    const active = activeStates.has(operation.state);
    if ((operation.kind === "enroll" || consoleMode()) && operation.target_account_id && !enrollmentTarget
      && operation.target_account_id !== authState().account?.account_id && isAdmin()) {
      enrollmentTarget = managedAccounts.find((account) => account.account_id === operation.target_account_id)
        || { account_id: operation.target_account_id, display_name: operation.target_display_name || operation.target_account_id, enabled: true };
      if (!accountsLoaded) void loadAccounts();
    }
    if (!active) cancelRequested = false;
    $("operation-panel").hidden = operation.id === dismissedOperationId;
    const observationLabel = observationModes[operation.observation_mode]?.label;
    const participantName = operation.target_display_name || enrollmentTarget?.display_name || "";
    $("operation-heading").textContent = operation.kind === "observe" ? `${observationLabel || "센서 검증"} · 표본 수집`
      : `${participantName ? `${participantName}님 · ` : ""}${operation.kind === "enroll" ? "얼굴 등록" : "얼굴 인증"}`;
    const guide = operation.guide || {};
    const panel = $("operation-panel");
    panel.dataset.state = active ? "busy" : operation.state === "cancelled" ? "cancelled"
      : operation.state === "succeeded" && resultLabel(operation)[2] === "good" ? "success" : "error";
    panel.dataset.phase = operation.state;
    panel.dataset.kind = operation.kind;
    panel.dataset.direction = ["C", "U", "D", "L", "R", "UL", "UR", "DL", "DR"].includes(guide.current)
      ? guide.current : "C";
    const coverage = ["enroll", "observe"].includes(operation.kind) && guide.coverage && typeof guide.coverage === "object"
      ? guide.coverage : null;
    $("pose-guide").hidden = !coverage || !active;
    $("operation-progress").hidden = !active;
    if (active) {
      badge("operation-state", operation.state === "processing" ? "확인 중" : "촬영 중", "waiting");
      const prompt = cancelRequested ? "촬영 취소를 확인하고 있습니다."
        : operation.state === "processing" ? "촬영한 정보를 확인하고 있어요"
        : operation.kind === "enroll" && guide.prompt === "고개를 천천히 돌려 원을 완성해 주세요"
          ? "화면을 보며 고개를 작은 원을 그리듯 천천히 움직여 주세요."
        : guide.prompt || operation.message || "정면을 바라봐 주세요.";
      if ($("operation-prompt").textContent !== prompt) $("operation-prompt").textContent = prompt;
      $("operation-description").textContent = operation.state === "processing"
        ? "잠시만 기다려 주세요. 곧 결과를 알려드릴게요." : operation.message || "카메라를 바라봐 주세요.";
    } else {
      const [label, prompt, tone] = resultLabel(operation);
      badge("operation-state", label, tone);
      $("operation-prompt").textContent = prompt;
      $("operation-description").textContent = operation.kind === "observe" && operation.result?.recorded === true
        ? "이 결과는 표본 수집 완료를 뜻합니다. 얼굴 인증이나 생체 판별의 검증 통과를 뜻하지 않습니다."
        : operation.message || operation.result?.reason || "";
    }
    let collected = 0;
    let required = 0;
    for (const item of document.querySelectorAll(".pose-grid [data-direction]")) {
      const values = coverage?.[item.dataset.direction];
      const have = Array.isArray(values) && finite(values[0]) ? Math.max(0, values[0]) : 0;
      const need = Array.isArray(values) && finite(values[1]) ? Math.max(0, values[1]) : 0;
      collected += Math.min(have, need);
      required += need;
      const done = need > 0 && have >= need;
      item.dataset.state = done ? "done" : guide.current === item.dataset.direction ? "current" : "waiting";
      item.querySelector("small").textContent = done ? "완료" : need ? `${have}/${need}` : "대기";
    }
    if (operation.state === "processing") {
      $("capture-progress").removeAttribute("value");
      $("progress-count").textContent = "";
    } else if (required > 0) {
      $("capture-progress").value = collected * 100 / required;
      $("progress-count").textContent = `${collected} / ${required}`;
    } else if (finite(guide.ring_frac) && guide.ring_frac >= 0 && guide.ring_frac <= 1) {
      $("capture-progress").value = guide.ring_frac * 100;
      $("progress-count").textContent = `${Math.round(guide.ring_frac * 100)}%`;
    } else {
      $("capture-progress").removeAttribute("value");
      $("progress-count").textContent = "";
    }
    $("progress-caption").textContent = operation.state === "processing" ? "결과 확인 중" : "촬영 진행";
    $("cancel-operation").hidden = !active;
    $("retry-operation").hidden = active || operation.state === "succeeded";
    $("dismiss-operation").hidden = active;
    $("dismiss-operation").textContent = operation.kind === "enroll" && enrollmentTarget ? "계정 관리로" : "닫기";
    if (!active && operation.kind === "enroll" && isAdmin() && refreshedOperationId !== operation.id) {
      refreshedOperationId = operation.id;
      void loadAccounts();
    }
    renderAuth();
  }

  async function authRequest(path, body, { controller = new AbortController(), timeoutMs = 10000 } = {}) {
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST", credentials: "same-origin", cache: "no-store",
        headers: body === undefined ? {} : { "Content-Type": "application/json", "X-PaceAuth-Action": "1" },
        body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
      });
      const data = response.status === 204 ? null : await response.json();
      if (!response.ok) {
        const error = new Error(typeof data?.detail === "string" ? data.detail : "요청을 처리할 수 없습니다.");
        error.status = response.status;
        throw error;
      }
      return data;
    } finally {
      clearTimeout(timeout);
    }
  }

  async function accountAction(kind) {
    if (authBusy || activeOperation() || operationUncertain || authState().enabled !== true) return;
    const creating = kind === "signup";
    if (!$("pilot-login-form").reportValidity()) return;
    const account = $("pilot-account").value.trim();
    const password = $("pilot-password").value;
    const displayName = $("pilot-display-name").value.trim();
    if (!/^[a-z0-9][a-z0-9_.-]{2,63}$/.test(account)) {
      feedback("계정 ID는 영문 소문자·숫자로 시작하는 3~64자의 영문 소문자·숫자·밑줄·마침표·하이픈으로 입력해 주세요.");
      return;
    }
    if (password.length < 10 || password.length > 128) {
      feedback("비밀번호는 10~128자로 입력해 주세요.");
      return;
    }
    authBusy = true;
    feedback(creating ? "체험 계정을 만들고 있습니다." : "계정을 확인하고 있습니다.", "waiting");
    renderAuth();
    try {
      await authRequest(`/api/auth/${kind}`, { account_id: account, password,
        ...(creating && displayName ? { display_name: displayName } : {}) });
      $("pilot-password").value = "";
      dismissedOperationId = null;
      feedback(creating ? "체험 계정을 만들었습니다. 생체정보 처리 안내를 확인해 주세요." : "로그인되었습니다.", "good");
      await pollStatus();
    } catch (error) {
      $("pilot-password").value = "";
      feedback(error.status === 401 ? "계정 ID 또는 비밀번호를 확인해 주세요."
        : error.status === 409 && creating ? "이미 사용 중인 계정 ID입니다. 로그인하거나 다른 ID를 사용해 주세요."
        : error.status === 429 ? "요청이 많습니다. 잠시 후 다시 시도해 주세요."
        : error.status ? error.message : "서버 응답을 확인하지 못했습니다. 연결 상태를 확인한 뒤 다시 로그인해 주세요.");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function saveProfile() {
    if (authBusy || activeOperation() || operationUncertain || authState().account?.authenticated !== true) return;
    if (!$("profile-form").reportValidity()) return;
    const displayName = $("profile-display-name").value.trim();
    if (!displayName || Array.from(displayName).length > 80) {
      feedback("표시할 이름을 1~80자로 입력해 주세요.");
      return;
    }
    authBusy = true;
    feedback("이름을 저장하고 있습니다.", "waiting");
    renderAuth();
    try {
      await authRequest("/api/auth/profile", { display_name: displayName });
      profileDirty = false;
      await pollStatus();
      feedback("표시 이름을 저장했습니다.", "good");
    } catch (error) {
      feedback(error.status ? error.message : "이름 저장 결과를 확인하지 못했습니다. 연결 상태를 확인해 주세요.");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function startOperation(kind, extra = {}) {
    if (!(kind === "observe" ? canObserve() : canStart()) || !["enroll", "authenticate", "observe"].includes(kind)) return;
    if (!consoleMode() && enrollmentTarget && kind !== "enroll") return;
    if (consoleMode() && kind === "authenticate" && enrollmentTarget?.face_status !== "active") return;
    const targetId = consoleMode() || kind === "enroll" ? enrollmentTarget?.account_id : null;
    const enrollmentOptions = kind === "enroll" ? {
      photo_consent: $("photo-consent").checked,
      re_enroll: needsReEnrollment(enrollmentTarget),
    } : {};
    authBusy = true;
    const random = crypto.getRandomValues(new Uint8Array(16));
    random[6] = (random[6] & 15) | 64;
    random[8] = (random[8] & 63) | 128;
    const hex = Array.from(random, (value) => value.toString(16).padStart(2, "0")).join("");
    pendingRequestId = `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    currentOperation = null;
    operationId = null;
    $("operation-panel").hidden = true;
    dismissedOperationId = null;
    lastOperationKind = kind;
    feedback("촬영 시작을 요청하고 있습니다.", "waiting");
    renderAuth();
    try {
      const operation = await authRequest(targetId && kind !== "observe" ? `/api/auth/accounts/${encodeURIComponent(targetId)}/${kind}` : `/api/auth/${kind}`, {
        consent: { version: authState().consent.version, agreed: true }, guide_mode: kind === "enroll" ? "circle" : "steps",
        client_request_id: pendingRequestId, ...(consoleMode() && kind === "observe" ? { target_account_id: targetId } : {}), ...enrollmentOptions, ...extra,
      });
      if (!operation?.id) throw new Error("Operation identifier missing");
      feedback("");
      renderOperation(operation);
      $("operation-panel").tabIndex = -1;
      $("operation-panel").focus({ preventScroll: true });
      (kind === "enroll" ? $("enrollment-header") : $("operation-panel")).scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
        block: kind === "enroll" ? "start" : "nearest",
      });
    } catch (error) {
      feedback(error.status ? error.message : "시작 요청의 결과를 확인하고 있습니다. 버튼을 다시 누르지 않아도 됩니다.");
      if (!error.status || error.status === 409 || error.status >= 500) {
        operationUncertain = true;
        void pollOperation(true);
      } else {
        pendingRequestId = null;
      }
      await pollStatus();
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function pollOperation(force = false) {
    if (operationBusy || document.hidden || !(consoleMode() ? authState().console_ready === true : authState().account?.authenticated === true)
      || (!force && !operationId && !operationUncertain)) return;
    operationBusy = true;
    lastOperationPollAt = performance.now();
    const controller = new AbortController();
    operationController = controller;
    try {
      const query = operationId ? `?id=${encodeURIComponent(operationId)}`
        : pendingRequestId ? `?client_request_id=${encodeURIComponent(pendingRequestId)}` : "";
      const operation = await authRequest(`/api/auth/operation${query}`,
        undefined, { controller, timeoutMs: 3500 });
      if (document.hidden) return;
      if (operation === null) {
        dismissedOperationId = authState().operation_id || dismissedOperationId;
        currentOperation = null;
        operationId = null;
        operationUncertain = false;
        cancelRequested = false;
        pendingRequestId = null;
        $("operation-panel").hidden = true;
        feedback("진행 중인 요청이 없습니다. 준비 상태를 확인한 뒤 다시 시작할 수 있습니다.");
        renderAuth();
      } else {
        renderOperation(operation, performance.now() - lastOperationPollAt);
      }
    } catch (error) {
      if (!document.hidden) {
        operationUncertain = true;
        feedback(error.status === 401 ? consoleMode() ? "페이지 연결을 갱신하고 있어요. 잠시만 기다려 주세요." : "계정 세션이 만료되었습니다. 다시 로그인해 주세요."
          : "작업 상태를 확인할 수 없습니다. 연결이 복구되면 결과를 다시 확인합니다.");
        renderAuth();
        if (error.status === 401) void pollStatus();
      }
    } finally {
      if (operationController === controller) operationController = null;
      operationBusy = false;
    }
  }

  async function cancelOperation() {
    if (!operationId || !activeOperation() || authBusy || cancelRequested || currentOperation.state === "processing") return;
    authBusy = true;
    cancelRequested = true;
    renderAuth();
    try {
      const operation = await authRequest("/api/auth/cancel", { id: operationId });
      if (operation?.id) renderOperation(operation);
      else await pollOperation(true);
    } catch (error) {
      cancelRequested = false;
      feedback(error.status === 409 ? "이미 서버에 제출된 요청입니다. 최종 결과를 기다려 주세요."
        : "취소 여부를 확인하고 있습니다. 서버 결과가 확인될 때까지 기다려 주세요.");
      operationUncertain = true;
      void pollOperation(true);
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function logout() {
    if (authBusy || activeOperation() || operationUncertain) return;
    authBusy = true;
    renderAuth();
    try {
      await authRequest("/api/auth/logout", {});
      currentOperation = null;
      operationId = null;
      dismissedOperationId = null;
      pendingRequestId = null;
      $("operation-panel").hidden = true;
      $("pilot-login-form").reset();
      accountMode = "login";
      $("biometric-consent").checked = false;
      $("photo-consent").checked = false;
      resetAccountManagement();
      feedback("로그아웃되었습니다.", "waiting");
      await pollStatus();
    } catch (error) {
      feedback(error.status === 409 ? "서버 처리 중입니다. 결과가 나온 뒤 로그아웃해 주세요." : "로그아웃을 확인하지 못했습니다. 다시 시도해 주세요.");
    } finally {
      authBusy = false;
      renderAuth();
    }
  }

  async function pollStatus() {
    if (statusBusy || document.hidden) return;
    statusBusy = true;
    const controller = new AbortController();
    statusController = controller;
    const timeout = setTimeout(() => controller.abort(), 3000);
    try {
      const response = await fetch("/api/status", { cache: "no-store", signal: controller.signal, credentials: "same-origin" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const status = await response.json();
      if (document.hidden) return;
      if (!status || !status.camera || !status.auth || !status.capture) throw new Error("Invalid status response");
      latestStatus = status;
      statusAt = Date.now();
      renderStatus(status);
      if (isStreaming()) schedulePreview();
    } catch {
      if (document.hidden) return;
      latestStatus = null;
      badge("camera-state", "서버 연결 끊김", "error");
      badge("auth-state", "확인 불가", "waiting");
      indicator("서버 연결 대기", "error");
      $("camera-message").textContent = "서버에 연결할 수 없습니다. 연결 상태를 자동으로 다시 확인합니다.";
      $("auth-description").textContent = "서버 연결 후 인증 서비스의 준비 상태를 확인할 수 있습니다.";
      $("auth-blockers").replaceChildren();
      $("auth-blockers").hidden = true;
      $("detail-server").textContent = "연결 끊김 · 재시도 중";
      $("detail-mode").textContent = "—";
      $("device-info").hidden = true;
      metrics();
      clearPreview("서버에 연결할 수 없어요", "네트워크와 PaceAuth 컨테이너의 실행 상태를 확인해 주세요.");
      renderAuth();
      badge("auth-state", "확인 불가", "waiting");
      $("auth-description").textContent = "서버 연결 후 인증 서비스의 준비 상태를 확인할 수 있습니다.";
    } finally {
      clearTimeout(timeout);
      if (statusController === controller) statusController = null;
      statusBusy = false;
    }
  }

  function enrollmentPreviewFinished() {
    return enrollmentMode && currentOperation?.kind === "enroll" && !activeStates.has(currentOperation.state);
  }

  function schedulePreview(delay = 0) {
    if (previewBusy || previewTimer !== null || document.hidden || !isStreaming() || enrollmentPreviewFinished()) return;
    previewTimer = setTimeout(() => {
      previewTimer = null;
      void pollPreview();
    }, delay);
  }

  function nextPaint(signal) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("Preview cancelled", "AbortError"));
        return;
      }
      const cancelled = () => {
        cancelAnimationFrame(frame);
        reject(new DOMException("Preview cancelled", "AbortError"));
      };
      const frame = requestAnimationFrame((now) => {
        signal.removeEventListener("abort", cancelled);
        resolve(now);
      });
      signal.addEventListener("abort", cancelled, { once: true });
    });
  }

  async function pollPreview() {
    if (previewBusy || document.hidden || !isStreaming() || enrollmentPreviewFinished()) return;
    previewBusy = true;
    const generation = previewGeneration;
    const view = selectedView;
    const controller = new AbortController();
    previewController = controller;
    const timeout = setTimeout(() => controller.abort(), 2500);
    let retryDelay = 0;
    let nextUrl = null;
    let nextImage = null;
    try {
      const query = lastSequence === null ? "" : `?after=${encodeURIComponent(lastSequence)}&wait_ms=1000`;
      const response = await fetch(`/api/preview/${view}.jpg${query}`, {
        cache: "no-store", signal: controller.signal, credentials: "same-origin",
      });
      const headersReceivedAt = performance.now();
      if (response.status === 204) return;
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const sequence = response.headers.get("X-Frame-Sequence");
      const ageHeader = response.headers.get("X-Frame-Age-Ms");
      if (!sequence || !/^\d+$/.test(sequence)) throw new Error("Missing frame sequence");
      if (ageHeader !== null && (!Number.isFinite(Number(ageHeader)) || Number(ageHeader) >= 3000)) {
        throw new Error("Stale preview frame");
      }
      if (sequence === lastSequence) {
        // Also tolerate an intermediary returning a duplicate response without decoding it.
        await response.body?.cancel();
        retryDelay = 16;
        return;
      }
      const blob = await response.blob();
      if (!blob.type.startsWith("image/")) throw new Error("Invalid preview response");
      nextUrl = URL.createObjectURL(blob);
      nextImage = new Image();
      nextImage.src = nextUrl;
      await nextImage.decode();
      const displayedAt = await nextPaint(controller.signal);
      if (ageHeader !== null && Number(ageHeader) + displayedAt - headersReceivedAt >= 3000) {
        throw new Error("Preview frame expired before display");
      }
      if (generation !== previewGeneration || view !== selectedView || document.hidden || !isStreaming() || enrollmentPreviewFinished()) return;
      const previousUrl = imageUrl;
      imageUrl = nextUrl;
      nextUrl = null;
      image.src = imageUrl;
      image.alt = views[view].alt;
      image.hidden = false;
      frameDisplayed = true;
      lastSequence = sequence;
      displayedCaptureAt = ageHeader !== null ? headersReceivedAt - Number(ageHeader) : null;
      displayStartedAt ??= displayedAt;
      displayTimes.push(displayedAt);
      $("preview-placeholder").hidden = true;
      $("preview-kind").textContent = views[view].label;
      $("preview-kind").hidden = false;
      indicator("실시간 영상", "good");
      renderFaces(displayedAt);
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    } catch {
      if (generation === previewGeneration && view === selectedView && isStreaming()) {
        retryDelay = 500;
        indicator("미리보기 수신 대기", "warning");
        clearPreview("선택한 영상을 받을 수 없어요", "영상 수신을 자동으로 다시 시도합니다. 다른 보기의 수신 상태도 확인할 수 있습니다.");
      }
    } finally {
      clearTimeout(timeout);
      if (nextImage) nextImage.removeAttribute("src");
      if (nextUrl) URL.revokeObjectURL(nextUrl);
      if (previewController === controller) previewController = null;
      previewBusy = false;
      schedulePreview(retryDelay);
    }
  }

  document.querySelectorAll(".view-button").forEach((button) => {
    button.addEventListener("click", () => {
      const view = button.dataset.view;
      if (view === selectedView || !views[view]) return;
      selectedView = view;
      document.querySelectorAll(".view-button").forEach((item) => {
        item.setAttribute("aria-pressed", String(item === button));
      });
      $("view-description").textContent = views[view].description;
      clearPreview("선택한 영상을 불러오고 있어요", "센서에서 영상이 도착하면 여기에 표시됩니다.");
      indicator("미리보기 준비 중", "waiting");
      if (latestStatus) renderStatus(latestStatus);
      schedulePreview();
    });
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      statusController?.abort();
      operationController?.abort();
      faceController?.abort();
      latestFaces = null;
      latestStatus = null;
      clearPreview("실시간 영상을 다시 확인합니다", "화면으로 돌아오면 최신 센서 상태를 확인합니다.");
    } else {
      void pollStatus();
    }
  });

  $("pilot-login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void accountAction(accountMode);
  });
  for (const mode of ["login", "signup"]) {
    $("account-mode-" + mode).addEventListener("click", () => {
      if (authBusy || activeOperation() || operationUncertain || authState().enabled !== true) return;
      accountMode = mode;
      feedback("");
      renderAuth();
    });
  }
  $("pilot-signup").addEventListener("click", () => void accountAction("signup"));
  $("pilot-logout").addEventListener("click", () => void logout());
  $("profile-display-name").addEventListener("input", () => { profileDirty = true; });
  $("profile-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void saveProfile();
  });
  $("biometric-consent").addEventListener("change", renderAuth);
  $("photo-consent").addEventListener("change", renderAuth);
  $("nav-live").addEventListener("click", () => setMainView("live"));
  $("nav-accounts").addEventListener("click", () => setMainView("accounts"));
  $("demo-account").addEventListener("change", () => {
    if (!consoleMode() || !canManage()) return;
    const account = managedAccounts.find((item) => item.account_id === $("demo-account").value && item.enabled !== false);
    selectParticipant(account || null);
    renderAuth();
  });
  $("console-add-account").addEventListener("click", () => {
    if (!canManage()) return;
    setMainView("accounts");
    $("new-account").click();
  });
  $("refresh-accounts").addEventListener("click", () => {
    accountFeedback("");
    void loadAccounts();
  });
  $("account-search").addEventListener("input", renderAccountManagement);
  $("new-account").addEventListener("click", () => {
    if (!canManage()) return;
    $("create-account-form").reset();
    accountFeedback("", "waiting", "create-account-feedback");
    $("create-account-dialog").showModal();
    $("new-display-name").focus();
  });
  $("close-create-account").addEventListener("click", () => {
    if (!canManage()) return;
    $("new-account-password").value = "";
    $("create-account-dialog").close();
  });
  $("create-account-dialog").addEventListener("cancel", (event) => {
    if (authBusy) event.preventDefault();
    else $("new-account-password").value = "";
  });
  $("create-account-form").addEventListener("submit", (event) => {
    event.preventDefault();
    void createManagedAccount();
  });
  $("managed-display-name").addEventListener("input", () => { managedProfileDirty = true; });
  $("managed-profile-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!$("managed-profile-form").reportValidity()) return;
    const name = $("managed-display-name").value.trim();
    if (!name || Array.from(name).length > 80) {
      accountFeedback("이름을 1~80자로 입력해 주세요.");
      return;
    }
    void updateManagedAccount("profile", { display_name: name });
  });
  $("toggle-account-enabled").addEventListener("click", () => {
    const account = managedAccount();
    if (account && !$("toggle-account-enabled").disabled) void updateManagedAccount("status", { enabled: account.enabled === false });
  });
  $("enroll-managed-account").addEventListener("click", prepareManagedEnrollment);
  $("clear-enrollment-target").addEventListener("click", () => {
    if (!canManage()) return;
    if (enrollmentTarget) selectedAccountId = enrollmentTarget.account_id;
    if (!consoleMode()) selectParticipant(null);
    renderAuth();
    setMainView("accounts");
  });
  $("delete-managed-account").addEventListener("click", () => openDeleteDialog("account"));
  $("delete-account-photo").addEventListener("click", () => openDeleteDialog("photo"));
  $("cancel-delete-account").addEventListener("click", () => {
    if (!canManage()) return;
    deletionTarget = null;
    $("delete-account-dialog").close();
  });
  $("delete-account-dialog").addEventListener("cancel", (event) => {
    if (authBusy) event.preventDefault();
    else deletionTarget = null;
  });
  $("confirm-delete-account").addEventListener("click", () => void deleteManagedAccount());
  $("start-enroll").addEventListener("click", () => void startOperation("enroll"));
  $("start-authenticate").addEventListener("click", () => void startOperation("authenticate"));
  $("cancel-operation").addEventListener("click", () => void cancelOperation());
  $("retry-operation").addEventListener("click", () => {
    if (lastOperationKind === "observe") $("observation-form").requestSubmit();
    else void startOperation(lastOperationKind);
  });
  $("dismiss-operation").addEventListener("click", () => {
    dismissedOperationId = operationId;
    $("operation-panel").hidden = true;
    if (lastOperationKind === "enroll" && enrollmentTarget) {
      selectedAccountId = enrollmentTarget.account_id;
      if (!consoleMode()) enrollmentTarget = null;
      $("biometric-consent").checked = false;
      $("photo-consent").checked = false;
      renderAuth();
      setMainView("accounts");
      return;
    }
    renderAuth();
    const returnButton = lastOperationKind === "observe" ? $("start-observation")
      : lastOperationKind === "enroll" ? $("start-enroll") : $("start-authenticate");
    if (!returnButton.disabled) returnButton.focus({ preventScroll: true });
  });
  $("observation-kind").addEventListener("change", renderAuth);
  $("observation-mode").addEventListener("change", renderAuth);
  $("observation-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!canObserve() || !$("observation-form").reportValidity()) return;
    if (!$("observation-subject").value.trim()
      || ($("observation-kind").value !== "live" && !$("observation-instrument").value.trim())) {
      feedback("대상 코드와 필요한 공격 매체 코드를 입력해 주세요.");
      return;
    }
    void startOperation("observe", {
      observation_mode: $("observation-mode").value,
      kind: $("observation-kind").value,
      split: $("observation-split").value,
      subject_id: $("observation-subject").value.trim(),
      instrument_id: $("observation-kind").value === "live" ? "" : $("observation-instrument").value.trim(),
      conditions: $("observation-conditions").value.trim(),
    });
  });

  // One preview request at a time waits for a new sequence; capture rate sets the pace.
  setInterval(pollStatus, 1000);
  // At most one recognition request is in flight. Slow inference never queues requests or delays preview.
  setInterval(() => {
    renderFaces();
    void pollFaces();
  }, 67);
  setInterval(() => {
    if (enrollmentMode) renderEnrollment();
    const interval = currentOperation?.state === "capturing" ? 100 : 500;
    if ((activeOperation() || operationUncertain) && performance.now() - lastOperationPollAt >= interval) void pollOperation();
  }, 100);
  setInterval(() => {
    displayMetrics();
    if (frameDisplayed && !isStreaming()) {
      indicator("최신 상태 확인 중", "warning");
      clearPreview("최신 센서 상태를 확인하고 있어요", "새 프레임이 수신되면 자동으로 다시 표시됩니다.");
    }
  }, 500);
  void pollStatus();
})();
