import { proxyJson } from "@/lib/backend";

/** Create a temporary phone-capture session (proxied to FastAPI). */
export async function POST() {
  try {
    return await proxyJson("/capture-session", { method: "POST" });
  } catch {
    return Response.json(
      { detail: "Could not start phone capture. Please try again." },
      { status: 502 },
    );
  }
}
