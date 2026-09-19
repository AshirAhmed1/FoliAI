import { getBackendBaseUrl } from "@/lib/backend";

type RouteContext = { params: Promise<{ sessionId: string }> };

/** Phone uploads an image into the temporary capture session (no inference). */
export async function POST(request: Request, context: RouteContext) {
  const { sessionId } = await context.params;
  try {
    const formData = await request.formData();
    const upstream = await fetch(
      `${getBackendBaseUrl()}/capture-session/${encodeURIComponent(sessionId)}/upload`,
      {
        method: "POST",
        body: formData,
      },
    );
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("Content-Type") || "application/json",
      },
    });
  } catch {
    return Response.json(
      { detail: "Photo upload failed. Please try again." },
      { status: 502 },
    );
  }
}
