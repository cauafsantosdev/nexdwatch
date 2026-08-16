export type TaskStatus = "queued" | "processing" | "completed" | "failed";

export interface TaskResult {
  user_id: number | null;
  logs_count: number;
}

export interface TaskError {
  code: string;
  message: string;
}

export interface ProfileSyncTask {
  task_id: string;
  type: "profile_sync";
  username: string;
  status: TaskStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  attempt: number;
  result: TaskResult | null;
  error: TaskError | null;
}
