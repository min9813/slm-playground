// ----------------------------------------------------------------------------
// Shared helpers
// ----------------------------------------------------------------------------
const toast = document.querySelector("#toast");
const runtimeStatus = document.querySelector("#runtimeStatus");
let toastTimer = 0;

function setToast(message) {
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
}

function describeMicError(error) {
  if (!window.isSecureContext) {
    return "マイクには安全なコンテキストが必要です。http://localhost:8000 か HTTPS で開いてください（リモートIPのHTTPは不可）。";
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return "このブラウザ/接続では getUserMedia が使えません（localhost か HTTPS で開いてください）。";
  }
  const name = error && error.name ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "マイクが拒否されました。アドレスバーのマイク許可を『許可』にして再読み込みしてください。";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "マイクが見つかりません。入力デバイスが接続されているか確認してください。";
  }
  if (name === "NotReadableError") {
    return "マイクを他のアプリが使用中の可能性があります。";
  }
  return `マイクエラー: ${name || (error && error.message) || "unknown"}`;
}

function seconds(value) {
  return `${Number(value || 0).toFixed(2)}s`;
}

function mb(value) {
  return `${Number(value || 0).toFixed(0)} MB`;
}

function syncPair(range, number) {
  range.addEventListener("input", () => (number.value = range.value));
  number.addEventListener("change", () => (range.value = number.value));
}

// ----------------------------------------------------------------------------
// Microphone recorder -> 16-bit PCM mono WAV (decoded client-side so the
// backend only ever receives plain WAV, no ffmpeg required).
// ----------------------------------------------------------------------------
class WavRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.stream = null;
    this.chunks = [];
    this.vad = null;
  }

  get isRecording() {
    return this.mediaRecorder?.state === "recording";
  }

  // opts.onSilence(): called once when speech is detected and then followed by
  //   `silenceMs` of quiet (auto-stop). Also fires after `maxMs` as a safety cap.
  // opts.threshold: RMS level (0..1) above which audio counts as speech.
  async start(opts = {}) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const error = new Error("getUserMedia unavailable");
      error.name = "InsecureContextError";
      throw error;
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(this.stream);
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    this.mediaRecorder.start();

    if (opts.onSilence) this._startVad(opts);
  }

  _startVad({ onSilence, silenceMs = 1200, threshold = 0.012, maxMs = 30000 }) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const context = new AudioCtx();
    const source = context.createMediaStreamSource(this.stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    const buffer = new Float32Array(analyser.fftSize);

    let raf = 0;
    let spoke = false;
    let silenceStart = 0;
    const startedAt = performance.now();
    let fired = false;
    const fire = () => {
      if (fired) return;
      fired = true;
      onSilence();
    };

    const tick = () => {
      if (!this.isRecording) return;
      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i += 1) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      const now = performance.now();

      if (rms > threshold) {
        spoke = true;
        silenceStart = 0;
      } else if (spoke) {
        if (!silenceStart) silenceStart = now;
        else if (now - silenceStart > silenceMs) return fire();
      }
      if (now - startedAt > maxMs) return fire();

      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    this.vad = { context, cancel: () => cancelAnimationFrame(raf) };
  }

  _stopVad() {
    if (!this.vad) return;
    this.vad.cancel();
    this.vad.context.close().catch(() => {});
    this.vad = null;
  }

  async stop() {
    this._stopVad();
    const recorder = this.mediaRecorder;
    if (!recorder) return null;
    const stopped = new Promise((resolve) => (recorder.onstop = resolve));
    recorder.stop();
    await stopped;
    this.stream.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.mediaRecorder = null;
    const blob = new Blob(this.chunks, { type: recorder.mimeType || "audio/webm" });
    return blobToWav(blob);
  }
}

async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioCtx();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  await audioContext.close();
  return encodeWav(audioBuffer);
}

function encodeWav(audioBuffer) {
  const length = audioBuffer.length;
  const sampleRate = audioBuffer.sampleRate;
  const channels = audioBuffer.numberOfChannels;

  // Downmix to mono.
  const mono = new Float32Array(length);
  for (let c = 0; c < channels; c += 1) {
    const data = audioBuffer.getChannelData(c);
    for (let i = 0; i < length; i += 1) mono[i] += data[i] / channels;
  }

  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, length * 2, true);

  let offset = 44;
  for (let i = 0; i < length; i += 1) {
    const sample = Math.max(-1, Math.min(1, mono[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

// ----------------------------------------------------------------------------
// Tabs
// ----------------------------------------------------------------------------
const tabs = Array.from(document.querySelectorAll(".tab"));
const modes = {
  tts: document.querySelector("#mode-tts"),
  asr: document.querySelector("#mode-asr"),
  chat: document.querySelector("#mode-chat"),
  vl: document.querySelector("#mode-vl"),
};

function selectMode(mode) {
  for (const tab of tabs) {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  }
  for (const [name, element] of Object.entries(modes)) {
    element.hidden = name !== mode;
  }
  if (mode === "vl") refreshModels();
}

tabs.forEach((tab) => tab.addEventListener("click", () => selectMode(tab.dataset.mode)));

// ----------------------------------------------------------------------------
// Status
// ----------------------------------------------------------------------------
function applyStatus(status) {
  runtimeStatus.textContent = status.loaded ? "model loaded" : "model lazy";
  ttsFields.runtimeDetail.textContent = `${status.device} / ${status.dtype}`;
  ttsFields.memoryDetail.textContent = status.cuda
    ? `${status.cuda.name} · ${mb(status.cuda.memory_used_mb)} used`
    : "cpu";
  if (status.realtime) {
    document.querySelector("#rtcBlock").hidden = false;
    document.querySelector("#rtcDivider").hidden = false;
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("status failed");
    applyStatus(await response.json());
  } catch {
    runtimeStatus.textContent = "offline";
  }
}

// ----------------------------------------------------------------------------
// TTS
// ----------------------------------------------------------------------------
const form = document.querySelector("#ttsForm");
const textInput = document.querySelector("#textInput");
const charCount = document.querySelector("#charCount");
const tokenRange = document.querySelector("#tokenRange");
const tokenNumber = document.querySelector("#tokenNumber");
const ttsButton = document.querySelector("#generateButton");
const audioPlayer = document.querySelector("#audioPlayer");
const downloadLink = document.querySelector("#downloadLink");
const filename = document.querySelector("#filename");
const canvas = document.querySelector("#waveform");

const ttsFields = {
  totalTime: document.querySelector("#totalTime"),
  generationTime: document.querySelector("#generationTime"),
  decodeTime: document.querySelector("#decodeTime"),
  loadTime: document.querySelector("#loadTime"),
  audioDuration: document.querySelector("#audioDuration"),
  audioTokens: document.querySelector("#audioTokens"),
  runtimeDetail: document.querySelector("#runtimeDetail"),
  memoryDetail: document.querySelector("#memoryDetail"),
};

let liveTimer = 0;
let liveStarted = 0;

function startLiveTimer() {
  liveStarted = performance.now();
  window.clearInterval(liveTimer);
  liveTimer = window.setInterval(() => {
    ttsFields.totalTime.textContent = seconds((performance.now() - liveStarted) / 1000);
  }, 90);
}

function stopLiveTimer() {
  window.clearInterval(liveTimer);
}

function setTtsBusy(isBusy) {
  ttsButton.disabled = isBusy;
  ttsButton.querySelector("span:last-child").textContent = isBusy ? "生成中" : "生成";
  runtimeStatus.textContent = isBusy ? "rendering" : "ready";
}

function applyTtsResult(result, clientSeconds) {
  const timings = result.timings;
  const cuda = result.runtime.cuda;

  ttsFields.totalTime.textContent = seconds(timings.total_seconds);
  ttsFields.generationTime.textContent = seconds(timings.generation_seconds);
  ttsFields.decodeTime.textContent = seconds(timings.decode_seconds);
  ttsFields.loadTime.textContent = seconds(timings.model_load_seconds);
  ttsFields.audioDuration.textContent = seconds(result.audio_duration_seconds);
  ttsFields.audioTokens.textContent = String(result.audio_tokens);
  ttsFields.runtimeDetail.textContent = `${result.runtime.device} / ${result.runtime.dtype}`;
  ttsFields.memoryDetail.textContent = cuda
    ? `${mb(cuda.memory_used_mb)} used / ${mb(cuda.memory_total_mb)}`
    : "cpu";

  const audioUrl = `${result.audio_url}?v=${Date.now()}`;
  audioPlayer.src = audioUrl;
  downloadLink.href = result.audio_url;
  downloadLink.download = result.filename;
  downloadLink.setAttribute("aria-disabled", "false");
  filename.textContent = `${result.filename} · ${Math.round(result.file_bytes / 1024)} KB · client ${seconds(clientSeconds)}`;
  drawWaveform(audioUrl).catch(() => drawIdleWave("waveform unavailable"));
}

async function drawWaveform(url) {
  const ctx = canvas.getContext("2d");
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const audioContext = new AudioContext();
  const decoded = await audioContext.decodeAudioData(buffer);
  const data = decoded.getChannelData(0);
  const { width, height } = canvas;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#17120e";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(210, 221, 43, 0.25)";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 24) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  ctx.strokeStyle = "#d2dd2b";
  ctx.lineWidth = 3;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / width));
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < data.length; i += 1) {
      const sample = data[start + i];
      min = Math.min(min, sample);
      max = Math.max(max, sample);
    }
    const y1 = ((1 - max) * height) / 2;
    const y2 = ((1 - min) * height) / 2;
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
  }
  ctx.stroke();
  await audioContext.close();
}

function drawIdleWave(label) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#17120e";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#009f9a";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let x = 0; x < width; x += 8) {
    const y = height / 2 + Math.sin(x / 24) * 20 + Math.sin(x / 7) * 5;
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "rgba(255, 250, 240, 0.64)";
  ctx.font = "16px monospace";
  ctx.fillText(label, 22, height - 24);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) {
    setToast("text required");
    return;
  }

  setTtsBusy(true);
  startLiveTimer();
  const clientStarted = performance.now();
  try {
    const response = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, max_new_tokens: Number(tokenNumber.value) }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "generation failed");
    applyTtsResult(payload, (performance.now() - clientStarted) / 1000);
  } catch (error) {
    setToast(error.message || "generation failed");
  } finally {
    stopLiveTimer();
    setTtsBusy(false);
  }
});

textInput.addEventListener("input", () => {
  charCount.textContent = `${textInput.value.length} / 2000`;
});
syncPair(tokenRange, tokenNumber);

// ----------------------------------------------------------------------------
// ASR
// ----------------------------------------------------------------------------
const asrRecord = document.querySelector("#asrRecord");
const asrFile = document.querySelector("#asrFile");
const asrPreview = document.querySelector("#asrPreview");
const asrSource = document.querySelector("#asrSource");
const asrRun = document.querySelector("#asrRun");
const asrTokens = document.querySelector("#asrTokens");
const asrTokensNum = document.querySelector("#asrTokensNum");
const asrText = document.querySelector("#asrText");
const asrMeta = document.querySelector("#asrMeta");
const asrCopy = document.querySelector("#asrCopy");

const asrRecorder = new WavRecorder();
let asrBlob = null;

function setAsrAudio(blob, label) {
  asrBlob = blob;
  asrPreview.src = URL.createObjectURL(blob);
  asrSource.textContent = label;
  asrRun.disabled = false;
}

asrRecord.addEventListener("click", async () => {
  if (asrRecorder.isRecording) {
    asrRecord.classList.remove("is-recording");
    asrRecord.querySelector(".rec-label").textContent = "録音開始";
    try {
      const wav = await asrRecorder.stop();
      setAsrAudio(wav, "recording");
    } catch (error) {
      setToast(error.message || "recording failed");
    }
    return;
  }
  try {
    await asrRecorder.start();
    asrRecord.classList.add("is-recording");
    asrRecord.querySelector(".rec-label").textContent = "停止";
  } catch (error) {
    console.error("microphone error", error);
    setToast(describeMicError(error));
  }
});

asrFile.addEventListener("change", async () => {
  const file = asrFile.files?.[0];
  if (!file) return;
  try {
    const wav = await blobToWav(file);
    setAsrAudio(wav, file.name);
  } catch {
    setToast("could not decode that audio file");
  }
});

asrRun.addEventListener("click", async () => {
  if (!asrBlob) {
    setToast("record or pick audio first");
    return;
  }
  asrRun.disabled = true;
  asrRun.querySelector("span:last-child").textContent = "認識中";
  asrText.textContent = "…";
  const clientStarted = performance.now();
  try {
    const fd = new FormData();
    fd.append("file", asrBlob, "audio.wav");
    fd.append("max_new_tokens", String(asrTokensNum.value));
    const response = await fetch("/api/asr", { method: "POST", body: fd });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "ASR failed");
    asrText.textContent = payload.text || "(empty)";
    asrCopy.setAttribute("aria-disabled", "false");
    const client = (performance.now() - clientStarted) / 1000;
    asrMeta.textContent =
      `${payload.text_tokens} tokens · gen ${seconds(payload.timings.generation_seconds)} · ` +
      `client ${seconds(client)} · in ${seconds(payload.audio_input_seconds)}`;
  } catch (error) {
    asrText.textContent = "認識に失敗しました。";
    setToast(error.message || "ASR failed");
  } finally {
    asrRun.disabled = false;
    asrRun.querySelector("span:last-child").textContent = "文字起こし";
  }
});

asrCopy.addEventListener("click", async () => {
  if (asrCopy.getAttribute("aria-disabled") === "true") return;
  try {
    await navigator.clipboard.writeText(asrText.textContent);
    setToast("copied");
  } catch {
    setToast("copy failed");
  }
});

syncPair(asrTokens, asrTokensNum);

// ----------------------------------------------------------------------------
// Voice chat (speech-to-speech)
// ----------------------------------------------------------------------------
const chatRecord = document.querySelector("#chatRecord");
const chatStatus = document.querySelector("#chatStatus");
const chatTokens = document.querySelector("#chatTokens");
const chatTokensNum = document.querySelector("#chatTokensNum");
const chatClear = document.querySelector("#chatClear");
const chatLog = document.querySelector("#chatLog");
const chatMeta = document.querySelector("#chatMeta");
const chatAutoStop = document.querySelector("#chatAutoStop");
const chatSilence = document.querySelector("#chatSilence");

const chatRecorder = new WavRecorder();
let chatStopping = false;

function clearChatEmpty() {
  const empty = chatLog.querySelector(".chat-empty");
  if (empty) empty.remove();
}

function addBubble(role, opts = {}) {
  clearChatEmpty();
  const bubble = document.createElement("div");
  bubble.className = `bubble bubble-${role}`;

  const tag = document.createElement("span");
  tag.className = "bubble-tag";
  tag.textContent = role === "user" ? "you" : "LFM2-Audio";
  bubble.appendChild(tag);

  if (opts.text !== undefined) {
    const text = document.createElement("p");
    text.className = "bubble-text";
    text.textContent = opts.text;
    bubble.appendChild(text);
  }
  if (opts.audioUrl) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = opts.audioUrl;
    if (opts.autoplay) audio.autoplay = true;
    bubble.appendChild(audio);
  }
  if (opts.meta) {
    const meta = document.createElement("span");
    meta.className = "bubble-meta";
    meta.textContent = opts.meta;
    bubble.appendChild(meta);
  }

  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

async function sendChat(wavBlob) {
  addBubble("user", { audioUrl: URL.createObjectURL(wavBlob) });
  const pending = addBubble("assistant", { text: "…" });
  const pendingText = pending.querySelector(".bubble-text");
  const clientStarted = performance.now();

  try {
    const fd = new FormData();
    fd.append("file", wavBlob, "audio.wav");
    fd.append("max_new_tokens", String(chatTokensNum.value));
    const response = await fetch("/api/chat", { method: "POST", body: fd });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "chat failed");

    pendingText.textContent = payload.text || "(音声のみ)";
    if (payload.audio_url) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.autoplay = true;
      audio.src = `${payload.audio_url}?v=${Date.now()}`;
      pending.appendChild(audio);
    }
    const client = (performance.now() - clientStarted) / 1000;
    const meta = document.createElement("span");
    meta.className = "bubble-meta";
    meta.textContent = `gen ${seconds(payload.timings.generation_seconds)} · reply ${seconds(payload.audio_duration_seconds)} · client ${seconds(client)}`;
    pending.appendChild(meta);
    chatLog.scrollTop = chatLog.scrollHeight;
  } catch (error) {
    pendingText.textContent = "返答の生成に失敗しました。";
    setToast(error.message || "chat failed");
  }
}

async function stopChatAndSend() {
  if (chatStopping || !chatRecorder.isRecording) return;
  chatStopping = true;
  chatRecord.classList.remove("is-recording");
  chatRecord.querySelector(".rec-label").textContent = "話しかける";
  chatStatus.textContent = "thinking";
  chatRecord.disabled = true;
  try {
    const wav = await chatRecorder.stop();
    await sendChat(wav);
  } catch (error) {
    setToast(error.message || "chat failed");
  } finally {
    chatRecord.disabled = false;
    chatStatus.textContent = "idle";
    chatStopping = false;
  }
}

chatRecord.addEventListener("click", async () => {
  if (chatRecorder.isRecording) {
    stopChatAndSend();
    return;
  }
  const autoStop = chatAutoStop.checked;
  try {
    await chatRecorder.start({
      onSilence: autoStop ? stopChatAndSend : null,
      silenceMs: Math.max(0.4, Number(chatSilence.value) || 1.2) * 1000,
    });
    chatRecord.classList.add("is-recording");
    chatRecord.querySelector(".rec-label").textContent = autoStop
      ? "聞き取り中…（無音で自動送信）"
      : "停止して送信";
    chatStatus.textContent = autoStop ? "listening (auto)" : "listening";
  } catch (error) {
    console.error("microphone error", error);
    setToast(describeMicError(error));
  }
});

chatClear.addEventListener("click", () => {
  chatLog.innerHTML = '<div class="chat-empty">まだ会話はありません。「話しかける」で開始します。</div>';
  chatMeta.textContent = "";
});

syncPair(chatTokens, chatTokensNum);

// ----------------------------------------------------------------------------
// Realtime voice chat (WebRTC via fastrtc)
// ----------------------------------------------------------------------------
const rtcBlock = document.querySelector("#rtcBlock");
const rtcDivider = document.querySelector("#rtcDivider");
const rtcToggle = document.querySelector("#rtcToggle");
const rtcAudio = document.querySelector("#rtcAudio");

const RTC_CONFIG = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

let pc = null;
let rtcEvents = null;
let rtcBubble = null;
let rtcText = "";

function rtcSetConnected(connected) {
  rtcToggle.classList.toggle("is-recording", connected);
  rtcToggle.querySelector(".rec-label").textContent = connected
    ? "切断する"
    : "リアルタイム会話を開始";
  chatStatus.textContent = connected ? "live" : "idle";
}

function rtcAppendText(content) {
  // A shorter/!prefix message means a new assistant turn started.
  if (!rtcBubble || content.length < rtcText.length || !content.startsWith(rtcText.slice(0, 3))) {
    rtcBubble = addBubble("assistant", { text: content });
  } else {
    rtcBubble.querySelector(".bubble-text").textContent = content;
  }
  rtcText = content;
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function rtcConnect() {
  let stream;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw Object.assign(new Error("insecure"), { name: "InsecureContextError" });
    }
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    setToast(describeMicError(error));
    return;
  }

  pc = new RTCPeerConnection(RTC_CONFIG);
  stream.getTracks().forEach((track) => pc.addTrack(track, stream));
  pc.createDataChannel("text");
  pc.addEventListener("track", (event) => {
    rtcAudio.srcObject = event.streams[0];
    rtcAudio.play().catch(() => {});
  });
  pc.addEventListener("connectionstatechange", () => {
    if (["failed", "closed", "disconnected"].includes(pc.connectionState)) rtcDisconnect();
  });

  try {
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") return resolve();
      const check = () => {
        if (pc.iceGatheringState === "complete") {
          pc.removeEventListener("icegatheringstatechange", check);
          resolve();
        }
      };
      pc.addEventListener("icegatheringstatechange", check);
      setTimeout(resolve, 2000); // fallback: send what we have
    });

    const webrtcId = Math.random().toString(36).substring(2);
    const response = await fetch("/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        webrtc_id: webrtcId,
      }),
    });
    if (!response.ok) throw new Error("offer rejected");
    await pc.setRemoteDescription(await response.json());

    rtcEvents = new EventSource(`/api/rtc/outputs?webrtc_id=${webrtcId}`);
    rtcEvents.addEventListener("output", (event) => {
      try {
        const data = JSON.parse(event.data);
        rtcAppendText(data.content ?? "");
      } catch {
        /* ignore malformed event */
      }
    });

    rtcBubble = null;
    rtcText = "";
    clearChatEmpty();
    rtcSetConnected(true);
  } catch (error) {
    setToast(error.message || "realtime connect failed");
    rtcDisconnect();
  }
}

function rtcDisconnect() {
  if (rtcEvents) {
    rtcEvents.close();
    rtcEvents = null;
  }
  if (pc) {
    pc.getSenders().forEach((s) => s.track && s.track.stop());
    pc.close();
    pc = null;
  }
  rtcAudio.srcObject = null;
  rtcSetConnected(false);
}

rtcToggle.addEventListener("click", () => {
  if (pc) rtcDisconnect();
  else rtcConnect();
});

// ----------------------------------------------------------------------------
// VL (image understanding) — model on/off rack + image inference
// ----------------------------------------------------------------------------
const modelRack = document.querySelector("#modelRack");
const vlGpu = document.querySelector("#vlGpu");
const vlFile = document.querySelector("#vlFile");
const vlImageName = document.querySelector("#vlImageName");
const vlPrompt = document.querySelector("#vlPrompt");
const vlGrounding = document.querySelector("#vlGrounding");
const vlTokens = document.querySelector("#vlTokens");
const vlTokensNum = document.querySelector("#vlTokensNum");
const vlRun = document.querySelector("#vlRun");
const vlCanvas = document.querySelector("#vlCanvas");
const vlText = document.querySelector("#vlText");
const vlMeta = document.querySelector("#vlMeta");
const vlCopy = document.querySelector("#vlCopy");
const vlActiveHint = document.querySelector("#vlActiveHint");
const vlFields = {
  total: document.querySelector("#vlTotalTime"),
  gen: document.querySelector("#vlGenTime"),
  prompt: document.querySelector("#vlPromptTime"),
  out: document.querySelector("#vlOutTokens"),
  boxes: document.querySelector("#vlBoxes"),
  runtime: document.querySelector("#vlRuntime"),
  memory: document.querySelector("#vlMemory"),
};

let vlImage = null; // HTMLImageElement of the current upload
let vlModels = []; // last fetched model registry
let vlBusy = false;

function gpuLabel(gpu) {
  if (!gpu) return "cpu";
  return `${mb(gpu.memory_used_mb)} / ${mb(gpu.memory_total_mb)} used`;
}

function activeVlKey() {
  // The currently-loaded VL model the run button targets (first one ON).
  const loaded = vlModels.find((m) => m.kind === "vl" && m.loaded);
  return loaded ? loaded.key : null;
}

function renderModels(payload) {
  vlModels = payload.models || [];
  vlGpu.textContent = `gpu ${gpuLabel(payload.gpu)}`;
  vlFields.memory.textContent = gpuLabel(payload.gpu);

  modelRack.innerHTML = "";
  for (const m of vlModels) {
    const row = document.createElement("div");
    row.className = `rack-row${m.loaded ? " is-on" : ""}`;

    const info = document.createElement("div");
    info.className = "rack-info";
    const name = document.createElement("strong");
    name.textContent = m.label + (m.grounding ? " · grounding" : "");
    const note = document.createElement("span");
    note.textContent = m.note || m.kind;
    info.append(name, note);

    const toggle = document.createElement("button");
    toggle.className = "rack-toggle";
    toggle.type = "button";
    toggle.textContent = m.loaded ? "ON" : "OFF";
    toggle.disabled = vlBusy;
    toggle.addEventListener("click", () => toggleModel(m.key, !m.loaded));

    row.append(info, toggle);
    modelRack.appendChild(row);
  }
  if (!vlModels.length) {
    modelRack.innerHTML = '<div class="rack-empty">no models</div>';
  }

  const key = activeVlKey();
  vlRun.disabled = vlBusy || !key || !vlImage;
  const target = vlModels.find((x) => x.key === key);
  vlActiveHint.textContent = key
    ? `推論対象: ${target.label}${target.grounding ? "（bbox対応）" : ""}`
    : "推論には対象のVLモデルをONにしてください。";
  vlGrounding.disabled = !target || !target.grounding;
  if (vlGrounding.disabled) vlGrounding.checked = false;
}

async function refreshModels() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("models failed");
    renderModels(await response.json());
  } catch {
    modelRack.innerHTML = '<div class="rack-empty">offline</div>';
  }
}

async function toggleModel(key, on) {
  vlBusy = true;
  modelRack.querySelectorAll(".rack-toggle").forEach((b) => (b.disabled = true));
  setToast(`${on ? "loading" : "unloading"} ${key}…`);
  try {
    const response = await fetch(`/api/models/${on ? "load" : "unload"}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "toggle failed");
    renderModels(payload);
    setToast(`${key} ${on ? "ON" : "OFF"}`);
  } catch (error) {
    setToast(error.message || "toggle failed");
    refreshModels();
  } finally {
    vlBusy = false;
  }
}

function drawVlImage(boxes = []) {
  const ctx = vlCanvas.getContext("2d");
  const { width, height } = vlCanvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#17120e";
  ctx.fillRect(0, 0, width, height);
  if (!vlImage) {
    ctx.fillStyle = "rgba(255, 250, 240, 0.64)";
    ctx.font = "16px monospace";
    ctx.fillText("no image", 22, height - 24);
    return;
  }
  // Fit the image into the canvas (letterbox) and remember the transform.
  const scale = Math.min(width / vlImage.width, height / vlImage.height);
  const drawW = vlImage.width * scale;
  const drawH = vlImage.height * scale;
  const offX = (width - drawW) / 2;
  const offY = (height - drawH) / 2;
  ctx.drawImage(vlImage, offX, offY, drawW, drawH);

  ctx.lineWidth = 3;
  ctx.font = "700 15px monospace";
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.bbox;
    const rx = offX + x1 * scale;
    const ry = offY + y1 * scale;
    const rw = (x2 - x1) * scale;
    const rh = (y2 - y1) * scale;
    ctx.strokeStyle = "#d2dd2b";
    ctx.strokeRect(rx, ry, rw, rh);
    if (box.label) {
      const text = box.label;
      const tw = ctx.measureText(text).width + 10;
      ctx.fillStyle = "#d2dd2b";
      ctx.fillRect(rx, Math.max(0, ry - 20), tw, 20);
      ctx.fillStyle = "#15120f";
      ctx.fillText(text, rx + 5, Math.max(13, ry - 6));
    }
  }
}

vlFile.addEventListener("change", () => {
  const file = vlFile.files?.[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    vlImage = img;
    vlImageName.textContent = `${file.name} · ${img.width}×${img.height}`;
    drawVlImage();
    vlRun.disabled = vlBusy || !activeVlKey();
  };
  img.onerror = () => setToast("could not load that image");
  img.src = URL.createObjectURL(file);
});

vlRun.addEventListener("click", async () => {
  const key = activeVlKey();
  if (!key) return setToast("VLモデルをONにしてください");
  if (!vlFile.files?.[0]) return setToast("画像を選択してください");

  vlBusy = true;
  vlRun.disabled = true;
  vlRun.querySelector("span:last-child").textContent = "推論中";
  vlText.textContent = "…";
  try {
    const fd = new FormData();
    fd.append("file", vlFile.files[0]);
    fd.append("key", key);
    fd.append("prompt", vlPrompt.value);
    fd.append("max_new_tokens", String(vlTokensNum.value));
    fd.append("grounding", String(vlGrounding.checked));
    const response = await fetch("/api/vl/infer", { method: "POST", body: fd });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "inference failed");

    vlText.textContent = payload.text || "(empty)";
    vlCopy.setAttribute("aria-disabled", "false");
    drawVlImage(payload.boxes || []);
    vlFields.total.textContent = seconds(payload.timings.total_seconds);
    vlFields.gen.textContent = seconds(payload.timings.generation_seconds);
    vlFields.prompt.textContent = seconds(payload.timings.prompt_seconds);
    vlFields.out.textContent = String(payload.output_tokens);
    vlFields.boxes.textContent = String((payload.boxes || []).length);
    vlFields.runtime.textContent = `${payload.runtime.device} / ${payload.runtime.dtype}`;
    vlFields.memory.textContent = gpuLabel(payload.gpu);
    if (payload.gpu) vlGpu.textContent = `gpu ${gpuLabel(payload.gpu)}`;
    vlMeta.textContent = `${payload.output_tokens} tok · in ${payload.input_tokens} tok · ${payload.image_size[0]}×${payload.image_size[1]}`;
  } catch (error) {
    vlText.textContent = "推論に失敗しました。";
    setToast(error.message || "inference failed");
  } finally {
    vlBusy = false;
    vlRun.disabled = !activeVlKey() || !vlImage;
    vlRun.querySelector("span:last-child").textContent = "推論";
  }
});

vlCopy.addEventListener("click", async () => {
  if (vlCopy.getAttribute("aria-disabled") === "true") return;
  try {
    await navigator.clipboard.writeText(vlText.textContent);
    setToast("copied");
  } catch {
    setToast("copy failed");
  }
});

syncPair(vlTokens, vlTokensNum);

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------
textInput.dispatchEvent(new Event("input"));
drawIdleWave("waiting for audio");
drawVlImage();
loadStatus();
