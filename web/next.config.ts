import type { NextConfig } from "next";

// 静态站：`pnpm build` 产出 out/，任意静态服务器可托管。部署在子路径下时设 BASE_PATH。
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: process.env.BASE_PATH || "",
  env: { NEXT_PUBLIC_BASE_PATH: process.env.BASE_PATH || "" },
  images: { unoptimized: true },
};

export default nextConfig;
