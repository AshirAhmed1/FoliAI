import { getBackendBaseUrl } from "@/lib/backend";

type RouteContext = { params: Promise<{ sessionId: string }> };

/** One-time image download for the desktop; backend marks session consumed. */
export async function GET(_request: Request, context: RouteContext) {
  const { sessionId } = await context.params;
  try {
    const upstream = await fetch(
      `${getBackendBaseUrl()}/capture-session/${encodeURIComponent(sessionId)}/image`,
      { cache: "no-store" },
    );

    if (!upstream.ok) {
      const body = await upstream.text();
      return new Response(body, {
        status: upstream.status,
        headers: {
          "Content-Type":
            upstream.headers.get("Content-Type") || "application/json",
        },
      });
    }

    const bytes = await upstream.arrayBuffer();
    return new Response(bytes, {
      status: 200,
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") || "image/jpeg",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json(
      { detail: "Could not retrieve the phone photo. Please try again." },
      { status: 502 },
    );
  }
}
