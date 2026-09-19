import { proxyJson } from "@/lib/backend";

type RouteContext = { params: Promise<{ sessionId: string }> };

/** Poll capture session status. */
export async function GET(_request: Request, context: RouteContext) {
  const { sessionId } = await context.params;
  try {
    return await proxyJson(`/capture-session/${encodeURIComponent(sessionId)}`);
  } catch {
    return Response.json(
      { detail: "Could not check phone capture status. Please try again." },
      { status: 502 },
    );
  }
}

/** Cancel / clean up a capture session. */
export async function DELETE(_request: Request, context: RouteContext) {
  const { sessionId } = await context.params;
  try {
    return await proxyJson(
      `/capture-session/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" },
    );
  } catch {
    return Response.json(
      { detail: "Could not cancel phone capture. Please try again." },
      { status: 502 },
    );
  }
}
