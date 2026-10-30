import type { Metadata } from "next";
import { Footer, Header } from "@/components/Shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "TGF Catalog",
  description: "Terrestrial gamma-ray flashes found by an event-based search of space-borne gamma-ray detectors, updated in near real time.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <Header />
        <main className="page pb-24">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
