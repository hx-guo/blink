export const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

/** "2017-06-23T02:00:05.918894976Z" → "2017-06-23 02:00:05.918895" */
export function utc(t: string, digits = 6): string {
  const [date, rest] = t.replace("Z", "").split("T");
  const [hms, frac = ""] = rest.split(".");
  return digits > 0 ? `${date} ${hms}.${frac.padEnd(9, "0").slice(0, digits)}` : `${date} ${hms}`;
}

export const displayName = (name: string) => `TGF ${name}`;

export function sci(x: number | null | undefined, digits = 1): string {
  if (x == null) return "–";
  if (x === 0) return "0";
  const [m, e] = x.toExponential(digits).split("e");
  return `${m}×10${superscript(Number(e))}`;
}

function superscript(n: number): string {
  const map: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
  return String(n).split("").map((c) => map[c] ?? c).join("");
}

export const lon = (x: number, digits = 2) => `${Math.abs(x).toFixed(digits)}°${x < 0 ? "W" : "E"}`;
export const lat = (x: number, digits = 2) => `${Math.abs(x).toFixed(digits)}°${x < 0 ? "S" : "N"}`;

export const years = (seconds: number) => seconds / (365.25 * 86400);

/** 事件页链接。静态托管下用查询参数；服务器配了重写时 `/tgf/<name>/` 也能打开 */
export function eventHref(e: { name: string | null; cand_id: string | null }): string {
  return e.name ? `/tgf/?name=${e.name}` : `/cand/?id=${e.cand_id}`;
}

export const instrumentHref = (slug: string) => `/instrument/?slug=${slug}`;
