import { useCallback, useState } from "react";

export interface Toast {
  id: number;
  kind: "error" | "success" | "info";
  message: string;
}

let nextId = 1;

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((all) => all.filter((t) => t.id !== id));
  }, []);

  const notify = useCallback(
    (kind: Toast["kind"], message: string) => {
      const id = nextId++;
      setToasts((all) => [...all.slice(-3), { id, kind, message }]);
      window.setTimeout(() => dismiss(id), kind === "error" ? 8000 : 4500);
    },
    [dismiss],
  );

  return { toasts, notify, dismiss };
}
