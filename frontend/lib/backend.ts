/** Server-side FastAPI base URL for Next.js route handlers (not exposed to the browser). */
export function getBackendBaseUrl(): string {
  return (
    process.env.BACKEND_URL?.replace(/\/$/, "") ||
    process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ||
    "http://127.0.0.1:8000"
  );
}

export async function proxyJson(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const url = `${getBackendBaseUrl()}${path}`;
  const upstream = await fetch(url, init);
  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") || "application/json",
    },
  });
}
