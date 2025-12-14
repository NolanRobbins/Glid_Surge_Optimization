import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  const body = await request.json()

  const baseUrl =
    process.env.NEXT_PUBLIC_GLID_API_BASE_URL ||
    process.env.GLID_API_BASE_URL ||
    'http://localhost:8000'

  const upstream = await fetch(`${baseUrl}/routes/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  })

  const text = await upstream.text()
  const contentType = upstream.headers.get('content-type') || 'application/json'

  return new NextResponse(text, {
    status: upstream.status,
    headers: { 'content-type': contentType },
  })
}

