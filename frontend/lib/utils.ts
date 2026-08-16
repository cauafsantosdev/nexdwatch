export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function normalizeUsername(value: string): string {
  return value.trim();
}
