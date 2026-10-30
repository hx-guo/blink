const SUP = "⁻⁰¹²³⁴⁵⁶⁷⁸⁹";
const PLAIN = "-0123456789";

/**
 * 字符串里的上标字符（sci() 的指数、yr⁻¹）按正文字体排成上标：正文字体的拉丁子集只有 ¹²³，
 * 其余上标字符会落到别的字体上，大小和基线都对不齐。HTML 里用 <sup>，SVG 的 <text> 里用 <tspan>。
 */
export default function Sup({ text, svg = false }: { text: string; svg?: boolean }) {
  const parts = text.split(/([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)/);
  return (
    <>
      {parts.map((p, i) => {
        if (i % 2 === 0) return p;
        const plain = [...p].map((c) => PLAIN[SUP.indexOf(c)]).join("").replace("-", "−");
        return svg ? (
          <tspan key={i} baselineShift="super" fontSize="0.72em">{plain}</tspan>
        ) : (
          <sup key={i} style={{ fontSize: "0.72em", lineHeight: 0 }}>{plain}</sup>
        );
      })}
    </>
  );
}
