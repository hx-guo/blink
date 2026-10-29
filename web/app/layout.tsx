import type { Metadata } from "next";
import { Source_Serif_4 } from "next/font/google";
import { Footer, Header } from "@/components/Shell";
import "./globals.css";

// 全站一种衬线字；构建时下载并随站点自托管，运行时不访问 Google
const serif = Source_Serif_4({ subsets: ["latin"], style: ["normal", "italic"], variable: "--font-serif" });

export const metadata: Metadata = {
  title: "TGF Catalog",
  description: "Terrestrial gamma-ray flashes found by an event-based search of space-borne gamma-ray detectors, updated in near real time.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={serif.variable} suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <Header />
        <main className="page py-10">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
