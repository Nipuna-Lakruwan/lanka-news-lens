(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const OUTLET = window.OUTLET_NAMES || {};
  let data, lang = "en";
  try { lang = localStorage.getItem("lnl-lang") || "en"; } catch (e) {}

  function render() {
    const day = data.days.find((d) => d.date === $("#day").value) || data.days[0];
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.l === lang)));
    $("#brief").innerHTML = day.stories.map((st, si) => {
      const b = st[lang] || st.en;
      const pid = (k) => `s${si}-${k}`;
      const sents = b.sentences.map((s) =>
        esc(s.text) + " " + s.cites.map((k) => `<sup class="c"><a href="#${pid(k)}">[${k}]</a></sup>`).join("")).join(" ");
      const langs = [...new Set(st.sources.map((s) => s.lang))].map((l) => `<span class="chip ${l}">${l}</span>`).join("");
      return `<article class="card brief"><h2 lang="${lang}" class="lang-${lang}">${esc(b.headline)}</h2>
        <div class="meta">${st.n_articles} articles · ${st.n_outlets} outlets ${langs}</div>
        <p lang="${lang}" class="lang-${lang}">${sents}</p>
        <details><summary>Sources (${st.sources.length})</summary><ol class="sources">${st.sources.map((s, k) =>
          `<li id="${pid(k + 1)}"><span class="chip ${s.lang}">${s.lang}</span>${esc(s.outlet)} — <a href="${esc(s.url)}" rel="noopener" lang="${s.lang}">${esc(s.title)}</a></li>`).join("")}</ol></details>
      </article>`;
    }).join("");
  }

  fetch("data/briefs.json").then((r) => r.json()).then((d) => {
    data = d;
    $("#day").innerHTML = d.days.map((x) => `<option value="${x.date}">${x.date} — ${esc(x.label || "")}</option>`).join("");
    $("#day").onchange = render;
    document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => {
      lang = b.dataset.l; try { localStorage.setItem("lnl-lang", lang); } catch (e) {} render();
    }));
    render();
  });
})();
