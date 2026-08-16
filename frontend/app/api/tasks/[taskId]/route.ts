import { fetchBackend, forwardJson, routeFailure } from "@/lib/api/server";

export async function GET(
  _request: Request,
  context: { params: Promise<{ taskId: string }> },
): Promise<Response> {
  try {
    const { taskId } = await context.params;
    if (!taskId) {
      return Response.json({ detail: "Task not found." }, { status: 404 });
    }
    return forwardJson(
      await fetchBackend(`/tasks/${encodeURIComponent(taskId)}`),
    );
  } catch (error) {
    return routeFailure(error);
  }
}
