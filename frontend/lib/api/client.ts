import type {
  LetterboxdImportResult,
  ProfileSyncSubmission,
} from "@/types/profile";
import type { RecommendationFeed } from "@/types/recommendations";
import type { ProfileSyncTask } from "@/types/tasks";

interface FastApiValidationIssue {
  msg?: unknown;
}

interface ErrorPayload {
  detail?: unknown;
  message?: unknown;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function messageFromPayload(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object") return fallback;
  const { detail, message } = payload as ErrorPayload;
  if (typeof detail === "string") return detail;
  if (typeof message === "string") return message;
  if (Array.isArray(detail)) {
    const issue = detail.find(
      (value): value is FastApiValidationIssue =>
        Boolean(value) && typeof value === "object" && "msg" in value,
    );
    if (typeof issue?.msg === "string") return issue.msg;
  }
  return fallback;
}

export async function apiFetch<T>(
  input: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch {
    throw new ApiError("NexdWatch is unreachable. Please try again.", 0);
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }
  if (!response.ok) {
    throw new ApiError(
      messageFromPayload(payload, "Something went wrong. Please try again."),
      response.status,
    );
  }
  return payload as T;
}

export function submitProfileSync(
  username: string,
): Promise<ProfileSyncSubmission> {
  return apiFetch("/api/profile-sync", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
}

export function getProfileSyncTask(taskId: string): Promise<ProfileSyncTask> {
  return apiFetch(`/api/tasks/${encodeURIComponent(taskId)}`);
}

export function importProfile(
  username: string,
  file: File,
): Promise<LetterboxdImportResult> {
  const formData = new FormData();
  formData.append("username", username);
  formData.append("file", file);
  return apiFetch("/api/profile-import", { method: "POST", body: formData });
}

export function getRecommendationFeed(userId: number): Promise<RecommendationFeed> {
  return apiFetch(`/api/recommendations/${userId}`);
}
