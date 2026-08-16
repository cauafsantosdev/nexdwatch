import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary: "brutal-button brutal-button-primary",
  secondary: "brutal-button brutal-button-secondary",
  ghost: "brutal-button brutal-button-ghost",
  danger: "brutal-button brutal-button-danger",
};

export function Button({
  className,
  variant = "primary",
  type = "button",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(variantClasses[variant], className)}
      {...props}
    />
  );
}
