// Checks the web calculator (docs/index.html) against 500 cases computed by the Python code.
// Run from the repository root:  node tests/test_calculator.js
const fs = require("fs"), path = require("path");
const html = fs.readFileSync(path.join(__dirname, "..", "docs", "index.html"), "utf8");
const a = html.indexOf("/* ===== THKFRE-CALC-BEGIN"), b = html.indexOf("/* ===== THKFRE-CALC-END");
const c = new Function(html.slice(a, b) + "\nreturn {thkfreRisk, kfreNonNA, kfreNA, kfreLP, ckdEpi2021};")();
const v = JSON.parse(fs.readFileSync(path.join(__dirname, "calculator_vectors.json"), "utf8"));
let m = 0;
for (const r of v) for (const [f, k] of [[c.thkfreRisk, "th"], [c.kfreNonNA, "nn"], [c.kfreNA, "na"]])
  m = Math.max(m, Math.abs(f(r.age, r.male, r.egfr, r.uacr) - r[k]));
const ex = c.thkfreRisk(60, 1, 20, 1000);
console.log(`500 cases: max |JS - Python| = ${m.toExponential(2)}; worked example TH-KFRE = ${(ex * 100).toFixed(1)}%`);
if (m > 1e-9 || Math.abs(ex - 0.240) > 0.0005) { console.log("FAIL"); process.exit(1); }
console.log("PASS");
