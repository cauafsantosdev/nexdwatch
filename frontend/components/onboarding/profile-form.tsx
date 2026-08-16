"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, Film, LoaderCircle, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

import { ImportForm } from "@/components/onboarding/import-form";
import { SyncProgress } from "@/components/onboarding/sync-progress";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ApiError,
  getProfileSyncTask,
  submitProfileSync,
} from "@/lib/api/client";
import { loadStoredProfile, saveStoredProfile } from "@/lib/storage";
import { normalizeUsername } from "@/lib/utils";
import type {
  LetterboxdImportResult,
  ProfileSyncSubmission,
  StoredProfile,
} from "@/types/profile";

const TASK_POLL_INTERVAL_MS = 1750;

export function ProfileForm() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [storedProfile, setStoredProfile] = useState<StoredProfile | null>(null);
  const [submission, setSubmission] = useState<ProfileSyncSubmission | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const handledTask = useRef<string | null>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setStoredProfile(loadStoredProfile()), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const syncMutation = useMutation({
    mutationFn: submitProfileSync,
    onSuccess: (result) => {
      handledTask.current = null;
      setSubmission(result);
    },
  });

  const taskQuery = useQuery({
    queryKey: ["profile-sync-task", submission?.task_id],
    queryFn: () => getProfileSyncTask(submission!.task_id),
    enabled: Boolean(submission?.task_id),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed"
        ? false
        : TASK_POLL_INTERVAL_MS;
    },
    retry: 1,
  });

  useEffect(() => {
    const task = taskQuery.data;
    if (task?.status !== "completed" || handledTask.current === task.task_id) return;
    if (!task.result?.user_id) {
      return;
    }
    handledTask.current = task.task_id;
    saveStoredProfile({
      userId: task.result.user_id,
      username: task.username,
    });
    router.push("/recommendations");
  }, [router, taskQuery.data]);

  function startSync(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const normalized = normalizeUsername(username);
    if (!normalized) {
      setValidationError("Enter your Letterboxd username.");
      return;
    }
    setUsername(normalized);
    setValidationError(null);
    syncMutation.mutate(normalized);
  }

  function openImport() {
    setSubmission(null);
    syncMutation.reset();
    setShowImport(true);
  }

  function handleImported(result: LetterboxdImportResult, normalized: string) {
    if (!result.user_id) return;
    saveStoredProfile({
      userId: result.user_id,
      username: normalized,
      importedCount: result.imported,
    });
    if (result.unresolved > 0) {
      toast.info(
        `${result.imported} films imported; ${result.unresolved} could not be matched to this catalog.`,
      );
    } else {
      toast.success(`${result.imported} films imported.`);
    }
    router.push("/recommendations");
  }

  if (submission) {
    const queryError =
      taskQuery.error instanceof ApiError ? taskQuery.error.message : undefined;
    const completionError =
      taskQuery.data?.status === "completed" && !taskQuery.data.result?.user_id
        ? "The profile loaded, but its recommendation identity was unavailable. Try the export fallback."
        : undefined;
    return (
      <SyncProgress
        username={submission.username}
        status={taskQuery.data?.status ?? submission.status}
        errorMessage={completionError || queryError}
        onRetry={() => {
          setSubmission(null);
          syncMutation.reset();
          syncMutation.mutate(submission.username);
        }}
        onImport={openImport}
      />
    );
  }

  return (
    <div>
      {storedProfile && !showImport && (
        <button
          type="button"
          className="continue-profile"
          onClick={() => router.push("/recommendations")}
        >
          <Film size={16} aria-hidden="true" /> Continue as @{storedProfile.username}
          <ArrowRight size={15} aria-hidden="true" />
        </button>
      )}

      {!showImport ? (
        <>
          <form onSubmit={startSync} className="profile-form" noValidate>
            <label htmlFor="letterboxd-username" className="field-label">
              Your Letterboxd username
            </label>
            <div className="profile-form-row">
              <div className="username-input-wrap">
                <span aria-hidden="true">@</span>
                <Input
                  id="letterboxd-username"
                  name="username"
                  autoComplete="username"
                  autoCapitalize="none"
                  spellCheck={false}
                  maxLength={15}
                  placeholder="username"
                  value={username}
                  disabled={syncMutation.isPending}
                  onChange={(event) => setUsername(event.target.value)}
                />
              </div>
              <Button type="submit" disabled={syncMutation.isPending} className="justify-center whitespace-nowrap">
                {syncMutation.isPending ? (
                  <><LoaderCircle className="motion-safe:animate-spin" size={18} aria-hidden="true" /> Starting…</>
                ) : (
                  <>Find my films <ArrowRight size={18} aria-hidden="true" /></>
                )}
              </Button>
            </div>
            {(validationError || syncMutation.error) && (
              <p className="inline-error" role="alert">
                {validationError ||
                  (syncMutation.error instanceof ApiError
                    ? syncMutation.error.message
                    : "Profile synchronization could not be started.")}
              </p>
            )}
          </form>

          <button type="button" className="import-trigger" onClick={() => setShowImport(true)}>
            <Upload size={17} aria-hidden="true" /> Import Letterboxd export
          </button>
        </>
      ) : (
        <ImportForm
          username={username}
          onUsernameChange={setUsername}
          onImported={handleImported}
          onClose={() => setShowImport(false)}
        />
      )}
    </div>
  );
}
