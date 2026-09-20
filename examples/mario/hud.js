// Injected into the game page by play.py.
//   __smb : bridge to the EmulatorJS instance (RAM out, buttons in, exact pause/step)
//   __hud : side panel showing what was sent to cloud Jev and what came back
(() => {
  const gm = () => window.EJS_emulator.gameManager;
  const PAD = [0, 3, 4, 5, 6, 7, 8];   // libretro ids: B, START, UP, DOWN, LEFT, RIGHT, A

  window.__smb = {
    ramOff: null,
    async ram() {
      let st = gm().getState();
      if (st && st.then) st = await st;
      const u = new Uint8Array(st);
      if (this.ramOff === null) {
        // fceumm save state: chunk tag "RAM\0" + uint32 size 0x800, then the 2 KB of NES RAM
        for (let i = 0; i < u.length - 8; i++) {
          if (u[i] === 0x52 && u[i + 1] === 0x41 && u[i + 2] === 0x4d && u[i + 3] === 0 &&
              u[i + 4] === 0 && u[i + 5] === 8 && u[i + 6] === 0 && u[i + 7] === 0) { this.ramOff = i + 8; break; }
        }
        if (this.ramOff === null) throw new Error("RAM chunk not found in save state");
      }
      return Array.from(u.subarray(this.ramOff, this.ramOff + 2048));
    },
    set(buttons) { for (const b of PAD) gm().simulateInput(0, b, buttons.includes(b) ? 1 : 0); },
    run(on) { gm().toggleMainLoop(on ? 1 : 0); },
    async step(n) {
      const f0 = gm().getFrameNum();
      this.run(true);
      while (gm().getFrameNum() - f0 < n) await new Promise(r => requestAnimationFrame(r));
      this.run(false);
      return gm().getFrameNum() - f0;
    },
    async act(buttons, n) {
      this.set(buttons);
      __hud.pad(buttons);
      const frames = await this.step(n);
      return { frames, ram: await this.ram() };
    },
  };

  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const paint = json => esc(json)
    .replace(/"([^"]+)":/g, '<span class="k">"$1"</span>:')
    .replace(/: ?"([^"]*)"/g, m => m.replace(/"([^"]*)"$/, '<span class="s">"$1"</span>'))
    .replace(/\b(-?\d+(?:\.\d+)?)\b(?![^<]*<\/span>)/g, '<span class="n">$1</span>')
    .replace(/\b(true|false|null)\b(?![^<]*<\/span>)/g, '<span class="b">$1</span>');

  const CSS = `
  #jevhud{position:fixed;z-index:2147483647;box-sizing:border-box;display:flex;flex-direction:column;
    background:#0a1120;color:#dbe4f0;font:10.5px/1.4 "Liberation Mono","DejaVu Sans Mono",monospace;
    border:1px solid #1c2a3f;overflow:hidden;text-align:left;box-shadow:-12px 0 0 #0a1120}
  #jevhud *{box-sizing:border-box}
  #jevhud .hd{display:flex;justify-content:space-between;align-items:baseline;padding:8px 12px 7px;
    border-bottom:1px solid #1c2a3f;background:#0d1729}
  #jevhud .hd b{font-size:12.5px;letter-spacing:.14em;color:#2dd4bf}
  #jevhud .hd span{color:#7d8da6}
  #jevhud .lab{display:flex;justify-content:space-between;padding:6px 12px 3px;color:#5f718d;
    letter-spacing:.12em;font-size:9.5px}
  #jevhud .lab i{font-style:normal;color:#7d8da6;letter-spacing:0}
  #jevhud pre{margin:0;padding:0 12px;flex:1 1 auto;min-height:0;overflow:hidden;white-space:pre-wrap;
    word-break:break-word;font:inherit;color:#aebbd0}
  #jevhud .k{color:#5eead4}#jevhud .s{color:#e2e8f0}#jevhud .n{color:#fbbf24}#jevhud .b{color:#c4b5fd}
  #jevhud .opts{padding:0 12px}
  #jevhud .o{display:grid;grid-template-columns:88px 1fr 40px;align-items:center;gap:8px;height:17px;color:#7d8da6}
  #jevhud .o .t{height:7px;background:#142033;position:relative}
  #jevhud .o .t u{position:absolute;inset:0 auto 0 0;background:#33486b}
  #jevhud .o em{font-style:normal;text-align:right}
  #jevhud .o.win{color:#ecfeff}#jevhud .o.win .t u{background:#14b8a6}#jevhud .o.win em{color:#2dd4bf}
  #jevhud .row{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-top:1px solid #1c2a3f;margin-top:6px}
  #jevhud .row div{padding:5px 0 5px 12px;border-right:1px solid #1c2a3f}
  #jevhud .row div:last-child{border-right:0}
  #jevhud .row small{display:block;color:#5f718d;font-size:9px;letter-spacing:.1em}
  #jevhud .row b{font-weight:400;color:#e2e8f0;font-size:11.5px}
  #jevhud .pad{display:flex;align-items:center;gap:6px;padding:6px 12px;border-top:1px solid #1c2a3f}
  #jevhud .pad span{color:#5f718d;font-size:9.5px;letter-spacing:.12em;margin-right:4px}
  #jevhud .pad kbd{font:inherit;min-width:24px;height:18px;line-height:17px;text-align:center;
    border:1px solid #24344e;color:#4d5f7c;border-radius:2px}
  #jevhud .pad kbd.on{background:#14b8a6;border-color:#14b8a6;color:#04211e}
  #jevhud .pad q{quotes:none;margin-left:auto;color:#e2e8f0}
  #jevhud .req{padding:0 12px 5px;color:#5f718d;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #jevhud .ft{border-top:1px solid #1c2a3f;background:#0d1729;padding:5px 12px;color:#7d8da6;
    white-space:nowrap;overflow:hidden}
  #jevhud .ft span{display:block}
  #jevhud .ft b{font-weight:400;color:#dbe4f0}
  #jevhud.busy .hd b::after{content:"  ● asking";color:#fbbf24;letter-spacing:0;font-size:10px}
  #jevhud .banner{padding:3px 12px;background:#14b8a6;color:#04211e;display:none}
  #jevhud .banner.bad{background:#f87171}`;

  const KEYS = [[6, "←"], [7, "→"], [0, "B"], [8, "A"]];

  window.__hud = {
    el: null,
    mount(width) {
      const g = document.querySelector("#game").getBoundingClientRect();
      const style = document.createElement("style");
      style.textContent = CSS;
      document.head.appendChild(style);
      const el = this.el = document.createElement("div");
      el.id = "jevhud";
      Object.assign(el.style, { left: Math.round(g.right) + 12 + "px", top: Math.round(g.top) + "px",
        width: width + "px", height: Math.round(g.height) + "px" });
      el.innerHTML = `
        <div class="hd"><b id="jh-brand">JEV · SYSTEM ONE</b><span id="jh-model">cloud</span></div>
        <div class="banner" id="jh-banner"></div>
        <div class="lab">STATE SENT<i id="jh-endpoint">POST api.typesafe.ai/v1/systemone</i></div>
        <pre id="jh-state">waiting for the game to hand over control…</pre>
        <div class="lab">ANSWER<i id="jh-q"></i></div>
        <div class="opts" id="jh-opts"></div>
        <div class="row">
          <div><small>LATENCY</small><b id="jh-lat">–</b></div>
          <div><small>CONFIDENCE</small><b id="jh-conf">–</b></div>
          <div><small>P(DANGER)</small><b id="jh-danger">–</b></div>
          <div><small>TOKENS IN/OUT</small><b id="jh-tok">–</b></div>
        </div>
        <div class="pad"><span>PAD</span>${KEYS.map(([i, l]) => `<kbd data-b="${i}">${l}</kbd>`).join("")}<q id="jh-act"></q></div>
        <div class="req" id="jh-req">&nbsp;</div>
        <div class="ft"><span id="jh-f1"></span><span id="jh-f2"></span></div>`;
      document.body.appendChild(el);
      const r = el.getBoundingClientRect();
      return { x: Math.round(g.left), y: Math.round(g.top), width: Math.round(r.right - g.left), height: Math.round(g.height) };
    },
    $(id) { return document.getElementById(id); },
    brand(title, endpoint, callsLabel) {
      this.$("jh-brand").textContent = title; this.$("jh-endpoint").textContent = endpoint; this.callsLabel = callsLabel;
    },
    pad(buttons) { this.el.querySelectorAll("kbd").forEach(k => k.classList.toggle("on", buttons.includes(+k.dataset.b))); },
    thinking(p) {
      this.el.classList.add("busy");
      this.$("jh-state").innerHTML = paint(p.stateText);
      this.$("jh-q").textContent = `choice · ${p.kind} · decision ${p.n}`;
    },
    decided(p) {
      this.el.classList.remove("busy");
      this.$("jh-model").textContent = p.model;
      this.$("jh-opts").innerHTML = p.options.map(o =>
        `<div class="o${o.chosen ? " win" : ""}"><span>${esc(o.name)}</span><div class="t"><u style="width:${Math.round(o.p * 100)}%"></u></div><em>${o.p.toFixed(2)}</em></div>`).join("");
      this.$("jh-lat").textContent = p.latency + " ms";
      this.$("jh-conf").textContent = p.confidence == null ? "–" : p.confidence.toFixed(2);
      this.$("jh-danger").textContent = p.danger == null ? "–" : p.danger.toFixed(2);
      this.$("jh-tok").textContent = `${p.usage.input_tokens}/${p.usage.output_tokens}`;
      this.$("jh-act").textContent = "→ " + p.choice;
      this.$("jh-req").textContent = p.requestId || "";
    },
    footer(f) {
      this.$("jh-f1").innerHTML = `GAME  attempt <b>${f.attempt}</b> · world <b>${f.world}</b> · x <b>${f.x}</b> · <b>${f.gameTime}s</b> played`;
      this.$("jh-f2").innerHTML = `${this.callsLabel || "JEV"}   <b>${f.calls}</b> calls · <b>${f.tokens}</b> tokens · avg <b>${f.avgLatency}</b> ms per call`;
    },
    banner(text, bad) {
      const b = this.$("jh-banner");
      b.textContent = text; b.style.display = text ? "block" : "none"; b.classList.toggle("bad", !!bad);
    },
  };
})();
