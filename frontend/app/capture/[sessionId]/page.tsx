"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import {
  isAllowedImageFile,
  MAX_UPLOAD_BYTES,
  resizeImageForUpload,
} from "@/lib/imageUpload";

type PageStatus =
  | "loading"
  | "ready"
  | "preview"
  | "sending"
  | "sent"
  | "error";

export default function PhoneCapturePage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params?.sessionId ?? "";

  const cameraInputRef = useRef<HTMLInputElement>(null);
  const galleryInputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);

  const [status, setStatus] = useState<PageStatus>("loading");
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const clearPreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
    setSelectedFile(null);
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setStatus("error");
      setError("Invalid capture link.");
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `/api/capture-session/${encodeURIComponent(sessionId)}`,
          { cache: "no-store" },
        );
        const data = await res.json().catch(() => null);
        if (cancelled) return;

        if (!res.ok) {
          setStatus("error");
          setError(
            typeof data?.detail === "string"
              ? data.detail
              : "This capture link is invalid or expired.",
          );
          return;
        }

        if (data.status === "waiting") {
          setStatus("ready");
          return;
        }

        if (data.status === "uploaded" || data.status === "consumed") {
          setStatus("error");
          setError(
            "This capture session was already used. Start a new phone capture from your computer.",
          );
          return;
        }

        setStatus("error");
        setError("This capture link is no longer available.");
      } catch {
        if (!cancelled) {
          setStatus("error");
          setError("Could not reach the server. Please try again.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const onPickFile = useCallback(
    async (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;

      if (!isAllowedImageFile(file)) {
        setError("Please choose a JPEG, PNG, WebP, or GIF image.");
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        setError("File too large. Maximum upload size is 10MB.");
        return;
      }

      setError(null);
      const compressed = await resizeImageForUpload(file);
      clearPreview();
      const url = URL.createObjectURL(compressed);
      previewUrlRef.current = url;
      setPreviewUrl(url);
      setSelectedFile(compressed);
      setStatus("preview");
    },
    [clearPreview],
  );

  const sendToComputer = useCallback(async () => {
    if (!selectedFile || !sessionId) return;
    setStatus("sending");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(
        `/api/capture-session/${encodeURIComponent(sessionId)}/upload`,
        {
          method: "POST",
          body: formData,
        },
      );
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        setStatus("preview");
        setError(
          typeof data?.detail === "string"
            ? data.detail
            : "Photo upload failed. Please try again.",
        );
        return;
      }

      clearPreview();
      setStatus("sent");
    } catch {
      setStatus("preview");
      setError("Photo upload failed. Please try again.");
    }
  }, [selectedFile, sessionId, clearPreview]);

  return (
    <div className="min-h-full bg-zinc-50 px-4 py-8 font-sans">
      <main className="mx-auto flex w-full max-w-md flex-col gap-5">
        <header>
          <p className="text-sm font-medium tracking-wide text-emerald-700">
            FoliAI
          </p>
          <h1 className="mt-1.5 text-2xl font-semibold tracking-tight text-zinc-900">
            Send a leaf photo to your computer
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-zinc-600">
            For best results, keep the leaf in focus and fill most of the frame.
          </p>
        </header>

        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          onChange={(e) => {
            void onPickFile(e.target.files);
            e.target.value = "";
          }}
        />
        <input
          ref={galleryInputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          tabIndex={-1}
          aria-hidden
          onChange={(e) => {
            void onPickFile(e.target.files);
            e.target.value = "";
          }}
        />

        {status === "loading" && (
          <p className="rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-600 shadow-sm">
            Checking capture link…
          </p>
        )}

        {error && (
          <div
            className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
            role="alert"
          >
            {error}
          </div>
        )}

        {(status === "ready" || status === "preview" || status === "sending") && (
          <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={previewUrl}
                alt="Selected leaf preview"
                className="mx-auto max-h-80 w-full rounded-xl border border-zinc-200 object-contain bg-zinc-50"
              />
            ) : (
              <p className="text-center text-sm text-zinc-500">
                Take a new photo or choose one from your library.
              </p>
            )}

            <div className="mt-4 flex flex-col gap-3">
              {!previewUrl && (
                <>
                  <button
                    type="button"
                    onClick={() => cameraInputRef.current?.click()}
                    className="inline-flex min-h-12 items-center justify-center rounded-xl bg-emerald-700 px-4 text-sm font-medium text-white hover:bg-emerald-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700"
                  >
                    Take photo
                  </button>
                  <button
                    type="button"
                    onClick={() => galleryInputRef.current?.click()}
                    className="inline-flex min-h-12 items-center justify-center rounded-xl border border-zinc-300 bg-white px-4 text-sm font-medium text-zinc-800 hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
                  >
                    Choose from photos
                  </button>
                </>
              )}

              {previewUrl && (
                <>
                  <button
                    type="button"
                    disabled={status === "sending"}
                    onClick={() => void sendToComputer()}
                    className="inline-flex min-h-12 items-center justify-center rounded-xl bg-emerald-700 px-4 text-sm font-medium text-white hover:bg-emerald-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {status === "sending" ? "Sending photo…" : "Send to computer"}
                  </button>
                  <button
                    type="button"
                    disabled={status === "sending"}
                    onClick={() => {
                      clearPreview();
                      setStatus("ready");
                      setError(null);
                    }}
                    className="inline-flex min-h-12 items-center justify-center rounded-xl border border-zinc-300 bg-white px-4 text-sm font-medium text-zinc-800 hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Retake / Choose another
                  </button>
                </>
              )}
            </div>
          </section>
        )}

        {status === "sent" && (
          <section
            className="rounded-2xl border border-emerald-200 bg-emerald-50 px-5 py-6 text-center shadow-sm"
            aria-live="polite"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
              Photo sent
            </p>
            <p className="mt-2 text-lg font-semibold text-zinc-900">
              Return to your computer to continue.
            </p>
            <p className="mt-2 text-sm text-zinc-600">
              FoliAI will analyze the leaf on your desktop session.
            </p>
          </section>
        )}
      </main>
    </div>
  );
}
