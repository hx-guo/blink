import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Footer, Header } from "@/components/Shell";
import "./globals.css";

// 设计稿字体；构建时下载并随站点自托管，运行时不访问 Google
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "TGFs Catalog",
  description: "Terrestrial gamma-ray flashes found by an event-based search of space-borne gamma-ray detectors, updated in near real time.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen antialiased">
        <Header />
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
