import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google"; // Use Inter for clean modern look
import Background from "@/components/ui/background";
import { Navbar } from "@/components/ui/navbar";

const font = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    title: "Aegis Financial",
    description: "Production-grade AI Financial Tools",
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className={`${font.className} min-h-screen antialiased overflow-x-hidden selection:bg-primary/20 selection:text-primary`}>
                <Background />
                <Navbar />
                <main className="container pt-32 pb-20 px-4 md:px-8 max-w-7xl mx-auto min-h-screen">
                    {children}
                </main>
            </body>
        </html>
    );
}
