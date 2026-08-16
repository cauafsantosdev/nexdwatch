import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type BadgeTone = "orange" | "green" | "blue" | "neutral";

export function Badge({
  className,
  tone = "neutral",
  ...props
}: HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return <span className={cn("nexd-badge", `nexd-badge-${tone}`, className)} {...props} />;
}
