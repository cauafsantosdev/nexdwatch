import { fetchBackend, forwardJson, routeFailure } from "@/lib/api/server";

export async function POST(request: Request): Promise<Response> {
  try {
    const payload: unknown = await request.json();
    const username =
      payload && typeof payload === "object" && "username" in payload
        ? String(payload.username).trim()
        : "";
    if (!username) {
      return Response.json({ detail: "Enter a Letterboxd username." }, { status: 422 });
    }
    const response = await fetchBackend(
      `/users/${encodeURIComponent(username)}/sync-logs`,
      { method: "POST" },
    );
    return forwardJson(response);
  } catch (error) {
    return routeFailure(error);
  }
}
