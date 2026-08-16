"use client";

import { FileArchive, RefreshCw, Upload, X } from "lucide-react";
import { useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface FileDropProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
}

export function FileDrop({ file, onFileChange, disabled = false }: FileDropProps) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function choose(candidate?: File) {
    if (candidate) onFileChange(candidate);
  }

  return (
    <div
      className={cn("file-drop", dragging && "file-drop-active", disabled && "opacity-60")}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setDragging(false);
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        if (!disabled) choose(event.dataTransfer.files[0]);
      }}
    >
      <input
        ref={inputRef}
        id={inputId}
        className="sr-only"
        type="file"
        accept=".zip,application/zip"
        disabled={disabled}
        onChange={(event) => choose(event.target.files?.[0])}
      />
      {file ? (
        <div className="flex w-full items-center gap-3 text-left">
          <span className="file-drop-icon"><FileArchive aria-hidden="true" /></span>
          <div className="min-w-0 flex-1">
            <p className="truncate font-semibold text-[var(--ink)]">{file.name}</p>
            <p className="mt-1 text-sm text-[var(--muted)]">
              {(file.size / 1024 / 1024).toFixed(1)} MB · ready to import
            </p>
          </div>
          <Button
            variant="ghost"
            className="h-10 px-3"
            aria-label="Remove selected export"
            disabled={disabled}
            onClick={() => {
              onFileChange(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
          >
            <X size={18} aria-hidden="true" />
          </Button>
        </div>
      ) : (
        <label htmlFor={inputId} className="flex w-full cursor-pointer flex-col items-center py-3 text-center">
          <span className="file-drop-icon mb-3"><Upload aria-hidden="true" /></span>
          <span className="font-semibold text-[var(--ink)]">Drop your Letterboxd ZIP here</span>
          <span className="mt-1 text-sm text-[var(--muted)]">or choose the original export file</span>
        </label>
      )}
      {file && (
        <button
          type="button"
          className="mt-3 inline-flex items-center gap-2 text-sm font-semibold text-[var(--blue)] underline-offset-4 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <RefreshCw size={14} aria-hidden="true" /> Replace file
        </button>
      )}
    </div>
  );
}
