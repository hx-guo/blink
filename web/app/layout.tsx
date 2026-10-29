import type { Metadata } from "next";
import { Inter, Source_Serif_4 } from "next/font/google";
import { Footer, Header } from "@/components/Shell";
import "./globals.css";

// 设计稿字体；构建时下载并随站点自托管，运行时不访问 Google
const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const serif = Source_Serif_4({ subsets: ["latin"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "TGF Catalog",
  description: "Terrestrial gamma-ray flashes found by an event-based search of space-borne gamma-ray detectors, updated in near real time.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${serif.variable}`} suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <div className="sheet">
          <Header />
          <main className="px-5 sm:px-10 py-8">{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
