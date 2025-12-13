import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Port Traffic & Planning',
  description: 'Port traffic analysis, visualization, and planning tools using real-time trade data',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

