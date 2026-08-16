"use client";

import { useMutation } from "@tanstack/react-query";
import { ArrowRight, LoaderCircle, X } from "lucide-react";
import { FormEvent, useState } from "react";

import { Button } from "@/components/ui/button";
import { FileDrop } from "@/components/ui/file-drop";
import { Input } from "@/components/ui/input";
import { ApiError, importProfile } from "@/lib/api/client";
import { normalizeUsername } from "@/lib/utils";
import type { LetterboxdImportResult } from "@/types/profile";

interface ImportFormProps {
  username: string;
  onUsernameChange: (username: string) => void;
  onImported: (result: LetterboxdImportResult, username: string) => void;
  onClose: () => void;
}

export function ImportForm({
  username,
  onUsernameChange,
  onImported,
  onClose,
}: ImportFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const mutation = useMutation({
    mutationFn: async ({ normalized, archive }: { normalized: string; archive: File }) => {
      const result = await importProfile(normalized, archive);
      if (!result.user_id) {
        throw new ApiError(
          "The import completed, but no profile identity was returned.",
          500,
        );
      }
      return result;
    },
    onSuccess: (result, variables) => onImported(result, variables.normalized),
  });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = normalizeUsername(username);
    if (!normalized) {
      setValidationError("Enter the Letterboxd username this export belongs to.");
      return;
    }
    if (!file) {
      setValidationError("Choose the original Letterboxd export ZIP.");
      return;
    }
    setValidationError(null);
    mutation.mutate({ normalized, archive: file });
  }

  const apiError = mutation.error instanceof ApiError ? mutation.error.message : null;

  return (
    <section className="import-panel" aria-labelledby="import-title">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow text-[var(--green)]">OFFLINE FALLBACK</p>
          <h2 id="import-title" className="mt-2 text-2xl font-bold text-[var(--ink)]">
            Import your Letterboxd export
          </h2>
          <p className="mt-2 max-w-lg text-sm leading-6 text-[var(--muted)]">
            Upload the untouched ZIP from Letterboxd settings. Your catalog is matched locally by the NexdWatch backend.
          </p>
        </div>
        <Button variant="ghost" className="h-10 px-3" onClick={onClose} aria-label="Close import form">
          <X size={18} aria-hidden="true" />
        </Button>
      </div>

      <form className="mt-7 space-y-5" onSubmit={handleSubmit}>
        <div>
          <label htmlFor="import-username" className="field-label">Letterboxd username</label>
          <Input
            id="import-username"
            name="username"
            autoComplete="username"
            maxLength={15}
            value={username}
            disabled={mutation.isPending}
            onChange={(event) => onUsernameChange(event.target.value)}
          />
        </div>
        <FileDrop file={file} onFileChange={setFile} disabled={mutation.isPending} />

        {(validationError || apiError) && (
          <p className="inline-error" role="alert">{validationError || apiError}</p>
        )}

        <Button type="submit" disabled={mutation.isPending} className="w-full justify-center sm:w-auto">
          {mutation.isPending ? (
            <><LoaderCircle className="motion-safe:animate-spin" size={18} aria-hidden="true" /> Importing profile…</>
          ) : (
            <>Import and find my films <ArrowRight size={18} aria-hidden="true" /></>
          )}
        </Button>
      </form>
    </section>
  );
}
