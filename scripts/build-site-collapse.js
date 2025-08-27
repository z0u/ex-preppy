(function () {
  if (window.__exCollapseCodeInitialized) return;
  window.__exCollapseCodeInitialized = true;

  function firstLineFrom(area) {
    try {
      var codeEl = area.querySelector(
        "pre, code, .cm-content, .CodeMirror, .highlight"
      );
      var raw = codeEl ? codeEl.textContent || "" : area.textContent || "";
      var line = raw
        .replace(/\r/g, "")
        .split("\n")
        .map(function (s) {
          return s.trim();
        })
        .find(function (s) {
          return s.length > 0;
        });
      if (!line) return "Show code";
      if (line.length > 120) line = line.slice(0, 117) + "...";
      return line;
    } catch (e) {
      return "Show code";
    }
  }

  /** Wrap code areas in details elements for collapsing */
  function wrap(area) {
    if (area.closest("details")) return;
    var details = document.createElement("details");
    details.className = "ex-collapse-code";
    var summary = document.createElement("summary");
    summary.className = "ex-collapse-summary";
    var prefix = document.createElement("span");
    prefix.className = "ex-collapse-prefix";
    prefix.textContent = "Code: ";
    summary.appendChild(prefix);
    var label = document.createElement("span");
    label.className = "ex-collapse-label";
    label.textContent = firstLineFrom(area);
    summary.appendChild(label);
    var parent = area.parentNode;
    if (!parent) return;
    parent.insertBefore(details, area);
    details.appendChild(summary);
    details.appendChild(area);
  }

  function init() {
    var nodes = document.querySelectorAll(".jp-CodeCell .jp-InputArea");
    nodes.forEach
      ? nodes.forEach(wrap)
      : Array.prototype.forEach.call(nodes, wrap);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
