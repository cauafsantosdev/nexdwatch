import { fetchBackend, forwardJson, routeFailure } from "@/lib/api/server";

export async function POST(request: Request): Promise<Response> {
  try {
    const incoming = await request.formData();
    const username = String(incoming.get("username") ?? "").trim();
    const file = incoming.get("file");
    if (!username) {
      return Response.json({ detail: "Enter a Letterboxd username." }, { status: 422 });
    }
    if (!(file instanceof File)) {
      return Response.json(
        { detail: "Choose an official Letterboxd export ZIP." },
        { status: 422 },
      );
    }

    const body = new FormData();
    body.append("file", file, file.name);
    const response = await fetchBackend(
      `/users/${encodeURIComponent(username)}/import`,
      { method: "POST", body },
    );
    return forwardJson(response);
  } catch (error) {
    return routeFailure(error);
  }
}
