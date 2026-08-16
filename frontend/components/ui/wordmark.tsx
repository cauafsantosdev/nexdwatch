import Link from "next/link";

import { BrandMark } from "@/components/ui/brand-mark";
import { cn } from "@/lib/utils";

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <Link
      href="/"
      className={cn("wordmark", compact && "wordmark-compact")}
      aria-label="NexdWatch home"
    >
      <BrandMark />
      <span>NEXD<span>WATCH</span></span>
    </Link>
  );
}
