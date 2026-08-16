import { fetchBackend, forwardJson, routeFailure } from "@/lib/api/server";

export async function GET(
  _request: Request,
  context: { params: Promise<{ userId: string }> },
): Promise<Response> {
  try {
    const { userId } = await context.params;
    const parsedUserId = Number(userId);
    if (!Number.isInteger(parsedUserId) || parsedUserId <= 0) {
      return Response.json({ detail: "Profile not found." }, { status: 404 });
    }
    return forwardJson(
      await fetchBackend(`/recommendations/${parsedUserId}/feed`),
    );
  } catch (error) {
    return routeFailure(error);
  }
}
