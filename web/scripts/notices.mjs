// 第三方声明：随站点发给访客的代码、数据与字体，各附版权声明与许可证全文（MIT、ISC、Apache、OFL
// 都要求随作品分发）。构建前运行，写 public/THIRD-PARTY-NOTICES.txt。
import { existsSync, readFileSync, readdirSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const pkg = JSON.parse(readFileSync(join(web, "package.json"), "utf8"));

// Next.js 只取进浏览器的那部分运行时；其余直接依赖连同它们的依赖全收
const NEXT_CLIENT = ["react", "react-dom", "scheduler", "@swc/helpers", "styled-jsx"];
const found = new Map();
function walk(name, from) {
  if (found.has(name)) return;
  // 同 Node 的查找：自 from 起逐级向上找 node_modules/<name>（不经 exports，d3 不导出 package.json）
  let dir = null;
  for (let d = from; ; d = dirname(d)) {
    const c = join(d, "node_modules", name);
    if (existsSync(join(c, "package.json"))) {
      dir = realpathSync(c);
      break;
    }
    if (dirname(d) === d) break;
  }
  if (!dir) throw new Error(`cannot find ${name} from ${from}`);
  const p = JSON.parse(readFileSync(join(dir, "package.json"), "utf8"));
  found.set(name, { dir, version: p.version, license: p.license, homepage: p.homepage ?? p.repository?.url ?? p.repository ?? "" });
  if (name === "next") NEXT_CLIENT.forEach((d) => walk(d, dir));
  else Object.keys(p.dependencies ?? {}).forEach((d) => walk(d, dir));
}
Object.keys(pkg.dependencies).forEach((d) => walk(d, web));

const licenseText = (dir) => {
  const f = readdirSync(dir).find((f) => /^(licen[cs]e|copying)(\.(md|txt))?$/i.test(f));
  return f ? readFileSync(join(dir, f), "utf8").trim() : null;
};

const out = [
  "Third-party notices",
  "",
  "This site includes the following software, data and fonts. Each is distributed under its own license, reproduced below.",
  "",
];
const rule = "-".repeat(78);
out.push(rule, "Source Serif 4 4.005 (font, unmodified WOFF2 from Adobe), SIL Open Font License 1.1", "https://github.com/adobe-fonts/source-serif", "", readFileSync(join(web, "licenses/SourceSerif4-LICENSE.md"), "utf8").trim(), "");
out.push(rule, "Natural Earth (coastlines, via world-atlas), public domain", "https://www.naturalearthdata.com/about/terms-of-use/", "");
for (const [name, p] of [...found].sort(([a], [b]) => a.localeCompare(b))) {
  const text = licenseText(p.dir);
  if (!p.license) throw new Error(`no license for ${name}@${p.version}`);
  // 没带许可证文件的包（如 client-only，空的标记包）只写 package.json 声明的许可证
  out.push(rule, `${name} ${p.version}, ${p.license}`, String(p.homepage).replace(/^git\+/, ""), "", text ?? `Licensed under ${p.license}; the package includes no license file.`, "");
}
const target = join(web, "public/THIRD-PARTY-NOTICES.txt");
writeFileSync(target, out.join("\n"));
console.log(`notices: ${found.size} packages -> ${target}${existsSync(target) ? "" : " (failed)"}`);
