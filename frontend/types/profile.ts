export interface StoredProfile {
  userId: number;
  username: string;
  importedCount?: number;
}

export interface ProfileSyncSubmission {
  task_id: string;
  username: string;
  status: "queued" | "processing" | "completed" | "failed";
  reused: boolean;
}

export interface UnresolvedImportFilm {
  name: string;
  year: number | null;
  uri: string;
  reason: string;
}

export interface LetterboxdImportResult {
  user_id: number | null;
  watched_in_export: number;
  rated_in_export: number;
  imported: number;
  unresolved: number;
  unresolved_sample: UnresolvedImportFilm[];
}
