(() => {
  "use strict";

  const API_BASE = "";

  const els = {
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    form: document.getElementById("vizForm"),
    promptInput: document.getElementById("vizPromptInput"),
    runButton: document.getElementById("vizRunButton"),
    modeToggleButtons: Array.from(document.querySelectorAll(".mode-toggle__option")),
    temperatureSlider: document.getElementById("vizTemperatureSlider"),
    temperatureValue: document.getElementById("vizTemperatureValue"),
    maxTokensSlider: document.getElementById("vizMaxTokensSlider"),
    maxTokensValue: document.getElementById("vizMaxTokensValue"),
    main: document.getElementById("vizMain"),
    emptyState: document.getElementById("vizEmptyState"),
    pipelineFlow: document.getElementById("pipelineFlow"),
    stageDetail: document.getElementById("stageDetail"),
    stepBackBtn: document.getElementById("stepBackBtn"),
    playPauseBtn: document.getElementById("playPauseBtn"),
    stepFwdBtn: document.getElementById("stepFwdBtn"),
    speedSlider: document.getElementById("speedSlider"),
    playbackLabel: document.getElementById("playbackLabel"),
    contextStrip: document.getElementById("contextStrip"),
    replyText: document.getElementById("replyText"),
    stepScrubber: document.getElementById("stepScrubber"),
  };

  const state = {
    mode: "sft",
    data: null,          // response from /api/generate-with-internals
    stageTypes: [],       // ['tokens','embeddings','layer-0',...,'logits','sampled']
    currentStepIndex: 0,
    currentStageIndex: 0,
    playing: false,
    timer: null,
  };

  // ------------------------------------------------------------------
  // Status
  // ------------------------------------------------------------------

  function setStatus(kind, text) {
    els.statusDot.className = `status-dot status-dot--${kind}`;
    els.statusText.textContent = text;
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const info = await response.json();
      setStatus("ok", `ready · ${info.device}`);
    } catch (err) {
      setStatus("error", "server unavailable");
      console.error(err);
    }
  }

  // ------------------------------------------------------------------
  // Running the model
  // ------------------------------------------------------------------

  async function runVisualization(message) {
    els.runButton.disabled = true;
    setStatus("pending", "running forward passes…");

    const payload = {
      message,
      mode: state.mode,
      max_new_tokens: Number(els.maxTokensSlider.value),
      temperature: Number(els.temperatureSlider.value),
    };

    try {
      const response = await fetch(`${API_BASE}/api/generate-with-internals`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      loadRun(data);
      setStatus("ok", "ready");
    } catch (err) {
      setStatus("error", err.message);
      console.error(err);
    } finally {
      els.runButton.disabled = false;
    }
  }

  function loadRun(data) {
    stopPlayback();

    state.data = data;
    state.stageTypes = buildStageTypes(data.num_layers);
    state.currentStepIndex = 0;
    state.currentStageIndex = 0;

    els.emptyState.hidden = true;
    els.main.hidden = false;

    buildPipelineFlow();
    buildStepScrubber();
    render();
  }

  function buildStageTypes(numLayers) {
    const types = ["tokens", "embeddings"];
    for (let i = 0; i < numLayers; i++) types.push(`layer-${i}`);
    types.push("logits", "sampled");
    return types;
  }

  // ------------------------------------------------------------------
  // Pipeline flow (the static row of boxes; only highlighting changes)
  // ------------------------------------------------------------------

  const STAGE_META = {
    tokens: { label: "Tokens", sub: "current sequence" },
    embeddings: { label: "Embed", sub: "token + position" },
    logits: { label: "LM head", sub: "final LN → logits" },
    sampled: { label: "Sample", sub: "pick next token" },
  };

  function stageMeta(stageType) {
    if (stageType.startsWith("layer-")) {
      const i = stageType.split("-")[1];
      return { label: `Block ${i}`, sub: "attn → FFN" };
    }
    return STAGE_META[stageType];
  }

  function buildPipelineFlow() {
    els.pipelineFlow.innerHTML = "";

    state.stageTypes.forEach((stageType, index) => {
      if (index > 0) {
        const arrow = document.createElement("span");
        arrow.className = "pipeline-flow__arrow";
        arrow.textContent = "→";
        els.pipelineFlow.appendChild(arrow);
      }

      const meta = stageMeta(stageType);
      const box = document.createElement("button");
      box.type = "button";
      box.className = "pipeline-flow__box";
      box.dataset.stageIndex = String(index);
      box.innerHTML = `
        <span class="pipeline-flow__box-label">${meta.label}</span>
        <span class="pipeline-flow__box-sub">${meta.sub}</span>
      `;
      box.addEventListener("click", () => {
        pausePlayback();
        state.currentStageIndex = index;
        render();
      });
      els.pipelineFlow.appendChild(box);
    });
  }

  function updatePipelineFlow() {
    const boxes = els.pipelineFlow.querySelectorAll(".pipeline-flow__box");
    boxes.forEach((box) => {
      const i = Number(box.dataset.stageIndex);
      box.classList.toggle("pipeline-flow__box--active", i === state.currentStageIndex);
      box.classList.toggle("pipeline-flow__box--done", i < state.currentStageIndex);
    });
  }

  // ------------------------------------------------------------------
  // Step scrubber
  // ------------------------------------------------------------------

  function buildStepScrubber() {
    els.stepScrubber.innerHTML = "";
    state.data.steps.forEach((step, index) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "step-dot";
      dot.textContent = String(index + 1);
      dot.title = `Jump to token ${index + 1}`;
      dot.addEventListener("click", () => {
        pausePlayback();
        state.currentStepIndex = index;
        state.currentStageIndex = 0;
        render();
      });
      els.stepScrubber.appendChild(dot);
    });
  }

  function updateStepScrubber() {
    const dots = els.stepScrubber.querySelectorAll(".step-dot");
    dots.forEach((dot, index) => {
      dot.classList.toggle("step-dot--active", index === state.currentStepIndex);
      dot.classList.toggle("step-dot--done", index < state.currentStepIndex);
    });
  }

  // ------------------------------------------------------------------
  // Rendering the current stage
  // ------------------------------------------------------------------

  function render() {
    if (!state.data) return;

    updatePipelineFlow();
    updateStepScrubber();
    renderStageDetail();
    renderContextAndReply();
    renderPlaybackLabel();
  }

  function currentStep() {
    return state.data.steps[state.currentStepIndex];
  }

  function currentStageType() {
    return state.stageTypes[state.currentStageIndex];
  }

  function renderStageDetail() {
    const stageType = currentStageType();
    const step = currentStep();
    els.stageDetail.innerHTML = "";

    if (stageType === "tokens") {
      renderTokensStage(step);
    } else if (stageType === "embeddings") {
      renderEmbeddingsStage(step);
    } else if (stageType.startsWith("layer-")) {
      const layerIndex = Number(stageType.split("-")[1]);
      renderAttentionStage(step, layerIndex);
    } else if (stageType === "logits") {
      renderLogitsStage(step);
    } else if (stageType === "sampled") {
      renderSampledStage(step);
    }
  }

  function addHeader(title, desc) {
    const h = document.createElement("h3");
    h.className = "stage-detail__title";
    h.textContent = title;
    els.stageDetail.appendChild(h);

    const p = document.createElement("p");
    p.className = "stage-detail__desc";
    p.textContent = desc;
    els.stageDetail.appendChild(p);
  }

  function renderTokensStage(step) {
    addHeader(
      `Token ${step.step_index + 1} of ${state.data.steps.length} — current sequence`,
      `${step.context_tokens.length} tokens go into this forward pass. The highlighted one was sampled in the previous step.`
    );

    const list = document.createElement("div");
    list.className = "token-list";
    step.context_tokens.forEach((tok, i) => {
      const chip = document.createElement("span");
      const isNewest = step.step_index > 0 && i === step.context_tokens.length - 1;
      chip.className = `token-chip${isNewest ? " token-chip--newest" : ""}`;
      chip.textContent = tok;
      list.appendChild(chip);
    });
    els.stageDetail.appendChild(list);
  }

  function renderEmbeddingsStage(step) {
    addHeader(
      "Token + position embeddings",
      "Each token becomes a vector (its meaning) plus a position vector (its place in the sequence), added together. Shown: the first 6 dimensions and the vector's overall magnitude."
    );

    const grid = document.createElement("div");
    grid.className = "embed-grid";

    step.embeddings.forEach((emb, i) => {
      const card = document.createElement("div");
      const isNewest = step.step_index > 0 && i === step.embeddings.length - 1;
      card.className = `embed-card${isNewest ? " embed-card--newest" : ""}`;

      const tokenLabel = document.createElement("div");
      tokenLabel.className = "embed-card__token";
      tokenLabel.textContent = step.context_tokens[i];
      card.appendChild(tokenLabel);

      const bars = document.createElement("div");
      bars.className = "embed-card__bars";
      const maxAbs = Math.max(0.5, ...emb.preview.map((v) => Math.abs(v)));
      emb.preview.forEach((v) => {
        const bar = document.createElement("div");
        const height = Math.max(2, (Math.abs(v) / maxAbs) * 26);
        bar.className = `embed-card__bar${v < 0 ? " embed-card__bar--neg" : ""}`;
        bar.style.height = `${height}px`;
        bars.appendChild(bar);
      });
      card.appendChild(bars);

      const norm = document.createElement("div");
      norm.className = "embed-card__norm";
      norm.textContent = `‖v‖ = ${emb.norm.toFixed(2)}`;
      card.appendChild(norm);

      grid.appendChild(card);
    });

    els.stageDetail.appendChild(grid);
  }

  function renderAttentionStage(step, layerIndex) {
    const layer = step.layers[layerIndex];
    addHeader(
      `Transformer block ${layerIndex} — attention`,
      `${layer.heads.length} heads, each deciding independently which earlier tokens to pull information from. Rows = query token, columns = key token. Hover a cell for the exact weight.`
    );

    const grid = document.createElement("div");
    grid.className = "attn-grid";

    layer.heads.forEach((matrix, headIndex) => {
      const wrap = document.createElement("div");
      wrap.className = "attn-head";

      const label = document.createElement("span");
      label.className = "attn-head__label";
      label.textContent = `Head ${headIndex}`;
      wrap.appendChild(label);

      const canvas = document.createElement("canvas");
      const size = Math.min(240, Math.max(60, matrix.length * 14));
      canvas.width = size;
      canvas.height = size;
      drawHeatmap(canvas, matrix);

      canvas.addEventListener("mousemove", (evt) => {
        const rect = canvas.getBoundingClientRect();
        const T = matrix.length;
        const col = Math.min(T - 1, Math.floor(((evt.clientX - rect.left) / rect.width) * T));
        const row = Math.min(T - 1, Math.floor(((evt.clientY - rect.top) / rect.height) * T));
        const weight = matrix[row][col];
        tooltip.textContent =
          `${step.context_tokens[row] ?? "?"} → ${step.context_tokens[col] ?? "?"} : ${weight.toFixed(3)}`;
      });
      canvas.addEventListener("mouseleave", () => {
        tooltip.textContent = "";
      });

      wrap.appendChild(canvas);
      grid.appendChild(wrap);
    });

    els.stageDetail.appendChild(grid);

    const tooltip = document.createElement("div");
    tooltip.className = "attn-tooltip";
    tooltip.textContent = "Hover a heatmap cell to see the exact attention weight.";
    els.stageDetail.appendChild(tooltip);
  }

  function drawHeatmap(canvas, matrix) {
    const T = matrix.length;
    const ctx = canvas.getContext("2d");
    const cell = canvas.width / T;

    // low -> "panel-raised" navy, high -> accent amber
    const low = [33, 40, 64];
    const high = [232, 163, 61];

    for (let row = 0; row < T; row++) {
      for (let col = 0; col < T; col++) {
        const v = Math.max(0, Math.min(1, matrix[row][col]));
        const r = low[0] + (high[0] - low[0]) * v;
        const g = low[1] + (high[1] - low[1]) * v;
        const b = low[2] + (high[2] - low[2]) * v;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(col * cell, row * cell, cell, cell);
      }
    }
  }

  function renderLogitsStage(step) {
    addHeader(
      "Final layer norm → LM head → logits",
      "The last position's vector is projected to a score for every vocabulary word, then turned into probabilities. Top candidates shown below."
    );

    const list = document.createElement("div");
    list.className = "logits-list";

    const maxProb = Math.max(...step.top_candidates.map((c) => c.probability), 0.001);
    let sampledShown = false;

    step.top_candidates.forEach((c) => {
      const isSampled = c.token === step.sampled_token;
      if (isSampled) sampledShown = true;

      const row = document.createElement("div");
      row.className = `logit-row${isSampled ? " logit-row--sampled" : ""}`;

      const token = document.createElement("span");
      token.className = "logit-row__token";
      token.textContent = c.token;

      const track = document.createElement("div");
      track.className = "logit-row__bar-track";
      const fill = document.createElement("div");
      fill.className = "logit-row__bar-fill";
      fill.style.width = `${(c.probability / maxProb) * 100}%`;
      track.appendChild(fill);

      const pct = document.createElement("span");
      pct.className = "logit-row__pct";
      pct.textContent = `${(c.probability * 100).toFixed(1)}%`;

      row.appendChild(token);
      row.appendChild(track);
      row.appendChild(pct);
      list.appendChild(row);
    });

    els.stageDetail.appendChild(list);

    if (!sampledShown) {
      const note = document.createElement("p");
      note.className = "stage-detail__desc";
      note.style.marginTop = "12px";
      note.textContent = `Sampled token "${step.sampled_token}" wasn't among the top candidates shown (temperature sampling can pick a lower-probability word).`;
      els.stageDetail.appendChild(note);
    }
  }

  function renderSampledStage(step) {
    addHeader(
      "Token sampled",
      "This token is appended to the sequence, and the whole forward pass repeats to produce the next one."
    );

    const reveal = document.createElement("div");
    reveal.className = "sampled-reveal";
    reveal.innerHTML = `
      <span class="sampled-reveal__token">${escapeHtml(step.sampled_token)}</span>
      <span class="sampled-reveal__note">
        token ${step.step_index + 1} of ${state.data.steps.length}
      </span>
    `;
    els.stageDetail.appendChild(reveal);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ------------------------------------------------------------------
  // Context strip + reply (build up as steps complete)
  // ------------------------------------------------------------------

  function renderContextAndReply() {
    const step0 = state.data.steps[0];
    const basePromptTokens = state.data.steps[0].context_tokens;

    const sampledSoFar = [];
    for (let i = 0; i < state.currentStepIndex; i++) {
      sampledSoFar.push(state.data.steps[i].sampled_token);
    }
    const atSampledStage = currentStageType() === "sampled";
    if (atSampledStage) {
      sampledSoFar.push(currentStep().sampled_token);
    }

    els.contextStrip.innerHTML = "";
    basePromptTokens.forEach((tok) => {
      const chip = document.createElement("span");
      chip.className = "token-chip";
      chip.textContent = tok;
      els.contextStrip.appendChild(chip);
    });
    sampledSoFar.forEach((tok, i) => {
      const chip = document.createElement("span");
      const isNewest = i === sampledSoFar.length - 1 && atSampledStage;
      chip.className = `token-chip${isNewest ? " token-chip--newest" : ""}`;
      chip.textContent = tok;
      els.contextStrip.appendChild(chip);
    });

    els.replyText.textContent = sampledSoFar.length ? sampledSoFar.join(" ") : "—";
    void step0;
  }

  function renderPlaybackLabel() {
    const stageType = currentStageType();
    const meta = stageMeta(stageType);
    els.playbackLabel.textContent =
      `Token ${state.currentStepIndex + 1}/${state.data.steps.length} · ${meta.label}`;
  }

  // ------------------------------------------------------------------
  // Playback (advances one stage at a time; wraps step -> step)
  // ------------------------------------------------------------------

  function advanceStage() {
    const stagesPerStep = state.stageTypes.length;

    if (state.currentStageIndex < stagesPerStep - 1) {
      state.currentStageIndex += 1;
    } else if (state.currentStepIndex < state.data.steps.length - 1) {
      state.currentStepIndex += 1;
      state.currentStageIndex = 0;
    } else {
      stopPlayback();
      return;
    }
    render();
  }

  function retreatStage() {
    if (state.currentStageIndex > 0) {
      state.currentStageIndex -= 1;
    } else if (state.currentStepIndex > 0) {
      state.currentStepIndex -= 1;
      state.currentStageIndex = state.stageTypes.length - 1;
    }
    render();
  }

  function startPlayback() {
    if (state.playing) return;
    state.playing = true;
    els.playPauseBtn.textContent = "⏸";
    scheduleNextTick();
  }

  function scheduleNextTick() {
    clearTimeout(state.timer);
    state.timer = setTimeout(() => {
      if (!state.playing) return;
      advanceStage();
      if (state.playing) scheduleNextTick();
    }, Number(els.speedSlider.value));
  }

  function pausePlayback() {
    state.playing = false;
    els.playPauseBtn.textContent = "▶";
    clearTimeout(state.timer);
  }

  function stopPlayback() {
    pausePlayback();
  }

  // ------------------------------------------------------------------
  // Event wiring
  // ------------------------------------------------------------------

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = els.promptInput.value.trim();
    if (!message) return;
    runVisualization(message);
  });

  els.modeToggleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      els.modeToggleButtons.forEach((b) =>
        b.classList.toggle("mode-toggle__option--active", b === button)
      );
    });
  });

  els.temperatureSlider.addEventListener("input", () => {
    const value = Number(els.temperatureSlider.value);
    els.temperatureValue.textContent = value === 0 ? "0.0 (greedy)" : value.toFixed(1);
  });

  els.maxTokensSlider.addEventListener("input", () => {
    els.maxTokensValue.textContent = els.maxTokensSlider.value;
  });

  els.playPauseBtn.addEventListener("click", () => {
    if (state.playing) {
      pausePlayback();
    } else {
      startPlayback();
    }
  });

  els.stepBackBtn.addEventListener("click", () => {
    pausePlayback();
    retreatStage();
  });

  els.stepFwdBtn.addEventListener("click", () => {
    pausePlayback();
    advanceStage();
  });

  els.speedSlider.addEventListener("input", () => {
    if (state.playing) scheduleNextTick();
  });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------

  checkHealth();
  els.promptInput.focus();
})();
