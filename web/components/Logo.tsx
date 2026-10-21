/** 站标：红色圆盘（地球上的闪光）加两条轨道与卫星 */
export default function Logo({ size = 64 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <circle cx="32" cy="32" r="17" fill="var(--accent)" />
      <ellipse cx="32" cy="32" rx="29" ry="12" fill="none" stroke="var(--text-primary)" strokeWidth="2.5" transform="rotate(-35 32 32)" />
      <ellipse cx="32" cy="32" rx="29" ry="12" fill="none" stroke="var(--text-primary)" strokeWidth="2.5" transform="rotate(35 32 32)" />
      <circle cx="9" cy="17" r="3.5" fill="var(--text-primary)" />
      <circle cx="55" cy="49" r="4" fill="var(--text-primary)" />
    </svg>
  );
}
