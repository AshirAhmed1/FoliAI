"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { QRCodeSVG } from "qrcode.react";

type SessionCreateResponse = {
  session_id: string;
  expires_at: number;
  expires_in_seconds?: number;
  status: string;
};

type SessionStatusResponse = {
  session_id: string;
  status: string;
  expires_at: number;
  expires_in_seconds?: number;
  has_image?: boolean;
};

function isLocalhostUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return host === "localhost" || host === "127.0.0.1" || host === "::1";
  } catch {
    return true;
  }
}

function getAppBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "");
  if (configured) return configured;
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  return "http://localhost:3000";
}

type PhoneHandoffModalProps = {
  open: boolean;
  onClose: () => void;
  onImageReceived: (file: File) => void;
};

export default function PhoneHandoffModal({
  open,
  onClose,
  onImageReceived,
}: PhoneHandoffModalProps) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [captureUrl, setCaptureUrl] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Starting…");
  const [error, setError] = useState<string | null>(null);
  const [expiresLabel, setExpiresLabel] = useState(
    "This link expires in 10 minutes.",
  );

  const pollAbortRef = useRef<AbortController | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const onCloseRef = useRef(onClose);
  const onImageReceivedRef = useRef(onImageReceived);
  onCloseRef.current = onClose;
  onImageReceivedRef.current = onImageReceived;

  const cleanupSession = useCallback(async (id: string | null) => {
    if (!id) return;
    try {
      await fetch(`/api/capture-session/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
    } catch {
      // Best-effort cleanup.
    }
  }, []);

  const stopPolling = useCallback(() => {
    pollAbortRef.current?.abort();
    pollAbortRef.current = null;
  }, []);

  const closeModal = useCallback(async () => {
    stopPolling();
    const id = sessionIdRef.current;
    sessionIdRef.current = null;
    setSessionId(null);
    setCaptureUrl(null);
    await cleanupSession(id);
    onCloseRef.current();
  }, [stopPolling, cleanupSession]);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;
    let fetchingImage = false;
    setError(null);
    setStatusText("Starting…");
    setCaptureUrl(null);
    setSessionId(null);
    sessionIdRef.current = null;

    const appBase = getAppBaseUrl();
    if (isLocalhostUrl(appBase)) {
      setError(
        "Phone capture requires a network-accessible app URL. Set NEXT_PUBLIC_APP_URL to this computer's local network address (for example http://192.168.x.x:3000), then restart the Next.js server.",
      );
      setStatusText("Configuration needed");
      return;
    }

    const abort = new AbortController();
    pollAbortRef.current = abort;

    (async () => {
      try {
        const createRes = await fetch("/api/capture-session", {
          method: "POST",
          signal: abort.signal,
        });
        const created = (await createRes.json().catch(() => null)) as
          | SessionCreateResponse
          | { detail?: string }
          | null;

        if (cancelled || abort.signal.aborted) return;

        if (!createRes.ok || !created || !("session_id" in created)) {
          setError(
            typeof created === "object" &&
              created &&
              "detail" in created &&
              typeof created.detail === "string"
              ? created.detail
              : "Could not start phone capture. Please try again.",
          );
          setStatusText("Failed to start");
          return;
        }

        sessionIdRef.current = created.session_id;
        setSessionId(created.session_id);
        const url = `${appBase}/capture/${created.session_id}`;
        setCaptureUrl(url);
        setExpiresLabel("This link expires in 10 minutes.");
        setStatusText("Waiting for phone…");

        while (!abort.signal.aborted && !cancelled) {
          await new Promise((r) => setTimeout(r, 1800));
          if (abort.signal.aborted || cancelled) break;

          const statusRes = await fetch(
            `/api/capture-session/${encodeURIComponent(created.session_id)}`,
            { signal: abort.signal, cache: "no-store" },
          );
          const statusBody = (await statusRes.json().catch(() => null)) as
            | SessionStatusResponse
            | { detail?: string }
            | null;

          if (cancelled || abort.signal.aborted) break;

          if (statusRes.status === 410) {
            setError(
              typeof statusBody === "object" &&
                statusBody &&
                "detail" in statusBody &&
                typeof statusBody.detail === "string"
                ? statusBody.detail
                : "This phone capture link expired. Start a new session.",
            );
            setStatusText("Expired");
            stopPolling();
            return;
          }

          if (!statusRes.ok || !statusBody || !("status" in statusBody)) {
            continue;
          }

          if (typeof statusBody.expires_in_seconds === "number") {
            const mins = Math.max(
              1,
              Math.ceil(statusBody.expires_in_seconds / 60),
            );
            setExpiresLabel(
              `This link expires in about ${mins} minute${mins === 1 ? "" : "s"}.`,
            );
          }

          if (statusBody.status === "uploaded" && !fetchingImage) {
            fetchingImage = true;
            setStatusText("Photo received");
            stopPolling();

            const imageRes = await fetch(
              `/api/capture-session/${encodeURIComponent(created.session_id)}/image`,
              { cache: "no-store" },
            );
            if (!imageRes.ok) {
              setError("Could not retrieve the phone photo. Please try again.");
              setStatusText("Failed");
              return;
            }

            const blob = await imageRes.blob();
            const file = new File([blob], "phone-capture.jpg", {
              type: blob.type || "image/jpeg",
            });

            sessionIdRef.current = null;
            setSessionId(null);
            onImageReceivedRef.current(file);
            onCloseRef.current();
            return;
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        if (!cancelled) {
          setError("Could not start phone capture. Please try again.");
          setStatusText("Failed to start");
        }
      }
    })();

    return () => {
      cancelled = true;
      abort.abort();
      const id = sessionIdRef.current;
      if (id) {
        sessionIdRef.current = null;
        void cleanupSession(id);
      }
    };
  }, [open, stopPolling, cleanupSession]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        void closeModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, closeModal]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/50 p-4"
      role="presentation"
      onClick={() => void closeModal()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="phone-handoff-title"
        className="w-full max-w-md rounded-2xl border border-zinc-200 bg-white p-5 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <h2
          id="phone-handoff-title"
          className="text-lg font-semibold text-zinc-900"
        >
          Use your phone camera
        </h2>
        <p className="mt-1.5 text-sm leading-relaxed text-zinc-600">
          Scan this QR code with your phone to take or upload a leaf photo.
        </p>

        {error && (
          <div
            className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2.5 text-sm text-amber-900"
            role="alert"
          >
            {error}
          </div>
        )}

        {captureUrl && !error && (
          <div className="mt-4 flex flex-col items-center gap-3">
            <div className="rounded-xl border border-zinc-200 bg-white p-3">
              <QRCodeSVG value={captureUrl} size={200} level="M" includeMargin />
            </div>
            <p className="break-all text-center text-xs text-zinc-500">
              {captureUrl}
            </p>
          </div>
        )}

        <p className="mt-4 text-sm text-zinc-600" aria-live="polite">
          {statusText}
        </p>
        <p className="mt-1 text-xs text-zinc-400">{expiresLabel}</p>

        <button
          type="button"
          onClick={() => void closeModal()}
          className="mt-5 inline-flex min-h-11 w-full items-center justify-center rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
