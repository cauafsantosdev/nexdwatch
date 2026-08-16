import type { StoredProfile } from "@/types/profile";

const PROFILE_STORAGE_KEY = "nexdwatch:last-profile:v1";

function isStoredProfile(value: unknown): value is StoredProfile {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<StoredProfile>;
  return (
    Number.isInteger(candidate.userId) &&
    Number(candidate.userId) > 0 &&
    typeof candidate.username === "string" &&
    candidate.username.trim().length > 0
  );
}

export function loadStoredProfile(): StoredProfile | null {
  if (typeof window === "undefined") return null;
  try {
    const serialized = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (!serialized) return null;
    const profile: unknown = JSON.parse(serialized);
    if (!isStoredProfile(profile)) {
      window.localStorage.removeItem(PROFILE_STORAGE_KEY);
      return null;
    }
    return {
      ...profile,
      username: profile.username.trim(),
    };
  } catch {
    window.localStorage.removeItem(PROFILE_STORAGE_KEY);
    return null;
  }
}

export function saveStoredProfile(profile: StoredProfile): void {
  if (typeof window === "undefined" || !isStoredProfile(profile)) return;
  window.localStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
}

export function clearStoredProfile(): void {
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(PROFILE_STORAGE_KEY);
  }
}
