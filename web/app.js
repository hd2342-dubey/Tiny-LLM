(() => {
  "use strict";

  const API_BASE = ""; // same-origin: FastAPI serves this file itself

  const els = {
    statusDot: document.getElementById("statusDot"),
    statusText: document.getElementById("statusText"),
    internalsToggle: document.getElementById("internalsToggle"),
    rail: document.getElementById("internalsRail"),
    chatStream: document.getElementById("chatStream"),
    composerForm: document.getElementById("composerForm"),
    promptInput: document.getElementById("promptInput"),
    sendButton: document.getElementById("sendButton"),
    statGrid: document.getElementById("statGrid"),
    numLayersLabel: document.getElementById("numLayersLabel"),
    modeToggleButtons: Array.from(document.querySelectorAll(".mode-toggle__option")),
    modeNote: document.getElementById("modeNote"),
    temperatureSlider: document.getElementById("temperatureSlider"),
    temperatureValue: document.getElementById("temperatureValue"),
    maxTokensSlider: document.getElementById("maxTokensSlider"),
    maxTokensValue: document.getElementById("maxTokensValue"),
    tokenPreview: document.getElementById("tokenPreview"),
  };

  const state = {
    mode: "sft",
    availableModes: [],
    isSending: false,
  };

  const MODE_NOTES = {
    sft: "Fine-tuned on 12 Q&A examples with response-only loss masking.",
    pretrained: "Next-token prediction only — no instruction-following behavior yet.",
  };

  // ------------------------------------------------------------------
  // Status + model info
  // ------------------------------------------------------------------

  function setStatus(kind, text) {
    els.statusDot.className = `status-dot status-dot--${kind}`;
    els.statusText.textContent = text;
  }

  async function loadModelInfo() {
    try {
      const response = await fetch(`${API_BASE}/api/model-info`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const info = await response.json();

      for (const [key, value] of Object.entries(info)) {
        const el = els.statGrid.querySelector(`[data-stat="${key}"]`);
        if (el) {
          el.textContent = typeof value === "number" ? value.toLocaleString() : value;
        }
      }
      els.numLayersLabel.textContent = info.num_layers;
      state.availableModes = info.available_modes || [];

      syncModeButtons();
      setStatus("ok", `ready · ${info.device}`);
    } catch (err) {
      setStatus("error", "model info unavailable");
      console.error(err);
    }
  }

  function syncModeButtons() {
    els.modeToggleButtons.forEach((button) => {
      const mode = button.dataset.mode;
      const available = state.availableModes.includes(mode);
      button.disabled = !available;
      button.title = available ? "" : "Checkpoint not found on the server";
    });
  }

  // ------------------------------------------------------------------
  // Chat rendering
  // ------------------------------------------------------------------

  function scrollToBottom() {
    els.chatStream.scrollTop = els.chatStream.scrollHeight;
  }

  function appendMessage({ role, text, meta, isError }) {
    const wrapper = document.createElement("div");
    wrapper.className = `msg msg--${role}${isError ? " msg--error" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "msg__avatar";
    avatar.textContent = role === "user" ? "U" : "Σ";

    const bubble = document.createElement("div");
    bubble.className = "msg__bubble";

    const p = document.createElement("p");
    p.textContent = text;
    bubble.appendChild(p);

    if (meta) {
      const metaEl = document.createElement("div");
      metaEl.className = "msg__meta";
      metaEl.textContent = meta;
      bubble.appendChild(metaEl);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    els.chatStream.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function appendTypingIndicator() {
    const wrapper = document.createElement("div");
    wrapper.className = "msg msg--assistant msg--typing";
    wrapper.innerHTML = `
      <div class="msg__avatar">Σ</div>
      <div class="msg__bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>`;
    els.chatStream.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  // ------------------------------------------------------------------
  // Token preview
  // ------------------------------------------------------------------

  function renderTokenPreview(tokens, containedUnknown) {
    els.tokenPreview.innerHTML = "";

    if (!tokens || tokens.length === 0) {
      els.tokenPreview.innerHTML =
        '<span class="token-preview__placeholder">Send a message to see how the tokenizer splits it.</span>';
      return;
    }

    tokens.forEach((token) => {
      const chip = document.createElement("span");
      const isUnk = token === "<UNK>";
      chip.className = `token-chip${isUnk ? " token-chip--unk" : ""}`;
      chip.textContent = token;
      els.tokenPreview.appendChild(chip);
    });
  }

  // ------------------------------------------------------------------
  // Sending a message
  // ------------------------------------------------------------------

  async function sendMessage(message) {
    state.isSending = true;
    els.sendButton.disabled = true;

    appendMessage({ role: "user", text: message });
    const typingEl = appendTypingIndicator();

    const payload = {
      message,
      mode: state.mode,
      max_new_tokens: Number(els.maxTokensSlider.value),
      temperature: Number(els.temperatureSlider.value),
    };

    try {
      const response = await fetch(`${API_BASE}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      typingEl.remove();

      appendMessage({
        role: "assistant",
        text: data.reply,
        meta: `${data.mode} · ${data.completion_tokens} tokens generated`,
      });

      renderTokenPreview(data.prompt_token_preview, data.contained_unknown_words);
    } catch (err) {
      typingEl.remove();
      appendMessage({
        role: "assistant",
        text: `Something went wrong: ${err.message}`,
        isError: true,
      });
    } finally {
      state.isSending = false;
      els.sendButton.disabled = false;
      els.promptInput.focus();
    }
  }

  // ------------------------------------------------------------------
  // Event wiring
  // ------------------------------------------------------------------

  els.composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const message = els.promptInput.value.trim();
    if (!message || state.isSending) return;

    els.promptInput.value = "";
    sendMessage(message);
  });

  els.modeToggleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) return;
      state.mode = button.dataset.mode;

      els.modeToggleButtons.forEach((b) =>
        b.classList.toggle("mode-toggle__option--active", b === button)
      );
      els.modeNote.textContent = MODE_NOTES[state.mode] || "";
    });
  });

  els.temperatureSlider.addEventListener("input", () => {
    const value = Number(els.temperatureSlider.value);
    els.temperatureValue.textContent = value === 0 ? "0.0 (greedy)" : value.toFixed(1);
  });

  els.maxTokensSlider.addEventListener("input", () => {
    els.maxTokensValue.textContent = els.maxTokensSlider.value;
  });

  els.internalsToggle.addEventListener("click", () => {
    const isOpen = els.rail.classList.toggle("rail--open");
    els.internalsToggle.setAttribute("aria-expanded", String(isOpen));
  });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------

  setStatus("pending", "connecting…");
  loadModelInfo();
  els.promptInput.focus();
})();
