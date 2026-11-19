(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const PAGE = 30;
  const cache = {};
  let idx, shown = PAGE, results = [];

  const norm = (s) => s.normalize("NFC").toLowerCase();
  const load = (m) => cache[m] || (cache[m] = fetch(`data/timeline/${m}.json`).then((r) => r.json()));

  function hl(text, q) {
    const t = esc(text);
    if (!q) return t;
    const i = norm(text).indexOf(q);
    if (i < 0) return t;
    return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
  }

  async function update() {
    const q = norm($("#q").value.trim());
    const month = $("#month").value, topic = $("#topic").value, lg = $("#langs").value;
    const months = q && !month ? idx.months.map((m) => m.m) : [month || idx.months[0].m];
    $("#count").textContent = "Loading…";
    const all = (await Promise.all(months.map(load))).flat();
    results = all.filter((s) =>
      (!topic || s.tp === topic) &&
      (!lg || (lg === "multi" ? s.l.length >= 2 : s.l.length === 3)) &&
      (!q || s.a.some((a) => norm(a[0]).includes(q))));
    results.sort((a, b) => (q ? b.a.length - a.a.length : 0) || (b.s < a.s ? -1 : 1));
    shown = PAGE;
    draw(q);
  }

  function draw(q) {
    $("#count").textContent = `${results.length.toLocaleString()} stories`;
    $("#list").innerHTML = results.slice(0, shown).map((s) => {
      const counts = { en: 0, si: 0, ta: 0 };
      s.a.forEach((a) => counts[a[2]]++);
      const bar = ["en", "si", "ta"].map((l) => `<i style="width:${100 * counts[l] / s.a.length}%;background:var(--${l})"></i>`).join("");
      const head = s.a.find((a) => a[2] === "en") || s.a[0];
      return `<article class="card story">
        <h2 lang="${head[2]}" class="lang-${head[2]}">${hl(head[0], q)}</h2>
        <div class="meta">${s.s}${s.e !== s.s ? " → " + s.e : ""} · ${s.a.length} articles · ${s.o} outlets · ${esc(s.tp.replace("_", " / "))}
          ${s.l.map((l) => `<span class="chip ${l}">${l}</span>`).join("")}</div>
        <div class="bar" aria-hidden="true">${bar}</div>
        <ul>${s.a.slice(0, 8).map((a) => `<li><span class="chip ${a[2]}">${a[2]}</span><a href="${esc(a[3])}" rel="noopener" lang="${a[2]}" class="lang-${a[2]}">${hl(a[0], q)}</a> <span class="meta">${esc(idx.outlets[a[1]] || a[1])}</span></li>`).join("")}
        ${s.a.length > 8 ? `<li class="meta">+ ${s.a.length - 8} more</li>` : ""}</ul></article>`;
    }).join("");
    $("#more").hidden = shown >= results.length;
  }

  fetch("data/timeline/index.json").then((r) => r.json()).then((d) => {
    idx = d;
    $("#month").innerHTML = `<option value="">Latest month (or all, when searching)</option>` +
      d.months.map((m) => `<option value="${m.m}">${m.m} (${m.n.toLocaleString()} stories)</option>`).join("");
    $("#topic").innerHTML += d.topics.map((t) => `<option value="${t}">${t.replace("_", " / ")}</option>`).join("");
    let timer;
    $("#q").oninput = () => { clearTimeout(timer); timer = setTimeout(update, 250); };
    ["#month", "#topic", "#langs"].forEach((s) => ($(s).onchange = update));
    $("#more").onclick = () => { shown += PAGE; draw(norm($("#q").value.trim())); };
    update();
  });
})();
