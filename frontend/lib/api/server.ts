import "server-only";

const DEFAULT_ERROR = "NexdWatch is temporarily unavailable.";

export class ServerConfigurationError extends Error {}

export function backendUrl(path: string): string {
  const baseUrl = process.env.NEXDWATCH_API_URL?.trim().replace(/\/$/, "");
  if (!baseUrl) {
    throw new ServerConfigurationError("NEXDWATCH_API_URL is not configured");
  }
  return `${baseUrl}${path}`;
}

export async function fetchBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  return fetch(backendUrl(path), { ...init, cache: "no-store" });
}

export async function forwardJson(response: Response): Promise<Response> {
  const body = await response.text();
  return new Response(body || null, {
    status: response.status,
    headers: { "Content-Type": "application/json" },
  });
}

export function routeFailure(error: unknown): Response {
  if (!(error instanceof ServerConfigurationError)) {
    console.error("BFF request failed", error);
  }
  return Response.json({ detail: DEFAULT_ERROR }, { status: 503 });
}
