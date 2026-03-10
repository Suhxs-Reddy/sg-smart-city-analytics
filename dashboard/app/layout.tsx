import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Singapore Smart City Analytics',
  description: 'Real-time traffic analytics for 90 Singapore LTA cameras',
  openGraph: {
    title: 'Singapore Smart City Analytics',
    description: 'Real-time traffic analytics for 90 Singapore LTA cameras',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-secondary text-white p-4 shadow-lg">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-2xl font-bold">🚦 SG Smart City Analytics</h1>
            <div className="flex gap-4">
              <a
                href="https://github.com/Suhxs-Reddy/sg-smart-city-analytics"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary transition"
              >
                GitHub
              </a>
              <a
                href={`${process.env.NEXT_PUBLIC_API_URL}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-primary transition"
              >
                API Docs
              </a>
            </div>
          </div>
        </nav>
        {children}
      </body>
    </html>
  )
}
