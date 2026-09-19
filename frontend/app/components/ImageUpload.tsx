"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type PredictionItem = {
  class_name: string;
  confidence: number;
};

type PredictionReason = "non_plant" | "low_confidence" | null;

type Prediction = {
  class_name: string | null;
  confidence: number | null;
  matched: boolean;
  input_valid: boolean;
  reason: PredictionReason;
  message: string | null;
  predictions: PredictionItem[];
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";
const PREDICT_URL = `${API_BASE}/predict`;

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10MB (matches backend)
const MAX_IMAGE_DIMENSION = 800;
const REQUEST_TIMEOUT_MS = 30_000;

function formatClassName(name: string) {
  let readable = name.replace(/_+/g, " ").replace(/\s+/g, " ").trim();
  readable = readable.replace(/^Pepper\s+bell\b/i, "Bell Pepper");
  readable = readable.replace(/^Tomato\s+Tomato\b/i, "Tomato");
  readable = readable.replace(/YellowLeaf/gi, "Yellow Leaf");
  readable = readable.replace(/\b\w/g, (char) => char.toUpperCase());
  return readable;
}

/** Accept standard uploads plus common camera MIME/extension fallbacks. */
function isAllowedImageFile(file: File): boolean {
  if (ACCEPTED_TYPES.includes(file.type)) return true;
  if (file.type.startsWith("image/") && file.type !== "image/svg+xml") {
    return true;
  }
  return !file.type && /\.(jpe?g|png|webp|gif)$/i.test(file.name);
}

async function resizeImageForUpload(file: File): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(
      1,
      MAX_IMAGE_DIMENSION / Math.max(bitmap.width, bitmap.height),
    );
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return file;
    }
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.85),
    );
    if (!blob) return file;

    const baseName = file.name.replace(/\.[^.]+$/, "") || "upload";
    return new File([blob], `${baseName}.jpg`, { type: "image/jpeg" });
  } catch {
    return file;
  }
}

function isPredictionPayload(payload: unknown): payload is Prediction {
  if (!payload || typeof payload !== "object") return false;
  const p = payload as Prediction;
  if (typeof p.matched !== "boolean") return false;
  if (typeof p.input_valid !== "boolean") return false;
  if (
    !(
      p.reason === null ||
      p.reason === "non_plant" ||
      p.reason === "low_confidence"
    )
  ) {
    return false;
  }
  if (!(p.class_name === null || typeof p.class_name === "string")) return false;
  if (!(p.confidence === null || typeof p.confidence === "number")) return false;
  if (!(p.message === null || typeof p.message === "string")) return false;
  if (!Array.isArray(p.predictions)) return false;
  if (p.reason === "non_plant") {
    return p.predictions.length === 0;
  }
  if (p.predictions.length === 0) return false;
  return p.predictions.every(
    (item) =>
      item &&
      typeof item.class_name === "string" &&
      typeof item.confidence === "number",
  );
}

/** Prefer native capture on phones/tablets; use in-page webcam on desktop. */
function shouldUseNativeCameraCapture(): boolean {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return false;
  }
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const multiTouch = navigator.maxTouchPoints > 1;
  const mobileUa = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  return coarsePointer || multiTouch || mobileUa;
}

function UploadIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      className={className}
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
      />
    </svg>
  );
}

function CameraIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      className={className}
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Z"
      />
    </svg>
  );
}

function ConfidenceList({
  items,
  accent,
  ariaLabel,
}: {
  items: PredictionItem[];
  accent: "emerald" | "amber";
  ariaLabel: string;
}) {
  const barClass = accent === "emerald" ? "bg-emerald-600" : "bg-amber-500";

  return (
    <ol className="flex flex-col gap-3" aria-label={ariaLabel}>
      {items.map((item, index) => {
        const pct = Math.max(0, Math.min(100, item.confidence * 100));
        return (
          <li key={`${item.class_name}-${index}`}>
            <div className="flex items-start justify-between gap-3 text-sm">
              <span className="min-w-0 break-words font-medium text-zinc-900">
                <span className="mr-2 text-zinc-400">{index + 1}.</span>
                {formatClassName(item.class_name)}
              </span>
              <span className="shrink-0 tabular-nums text-zinc-600">
                {pct.toFixed(1)}%
              </span>
            </div>
            <div
              className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-white/90 ring-1 ring-zinc-200/70"
              role="progressbar"
              aria-valuenow={Math.round(pct)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${formatClassName(item.class_name)} confidence`}
            >
              <div
                className={`h-full rounded-full transition-[width] duration-300 ease-out ${barClass}`}
                style={{ width: `${pct}%` }}
              />
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export default function ImageUpload() {
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureButtonRef = useRef<HTMLButtonElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const handleSelectedFileRef = useRef<(file: File) => Promise<void>>(
    async () => undefined,
  );

  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isWebcamOpen, setIsWebcamOpen] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [isStartingCamera, setIsStartingCamera] = useState(false);

  const stopCamera = useCallback(() => {
    const stream = streamRef.current;
    if (stream) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsVideoReady(false);
  }, []);

  const closeWebcam = useCallback(() => {
    stopCamera();
    setIsWebcamOpen(false);
    setIsStartingCamera(false);
  }, [stopCamera]);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      const stream = streamRef.current;
      if (stream) {
        for (const track of stream.getTracks()) {
          track.stop();
        }
        streamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!isWebcamOpen) return;

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeWebcam();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isWebcamOpen, closeWebcam]);

  useEffect(() => {
    if (isWebcamOpen && isVideoReady) {
      captureButtonRef.current?.focus();
    }
  }, [isWebcamOpen, isVideoReady]);

  // Start the MediaStream only after the modal (and <video>) have mounted.
  useEffect(() => {
    if (!isWebcamOpen) return;

    let cancelled = false;
    setIsStartingCamera(true);
    setIsVideoReady(false);

    async function initCamera() {
      try {
        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false,
          });
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        }

        if (cancelled) {
          for (const track of stream.getTracks()) {
            track.stop();
          }
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (!video) {
          for (const track of stream.getTracks()) {
            track.stop();
          }
          streamRef.current = null;
          setIsWebcamOpen(false);
          setError(
            "Camera access was unavailable. You can still upload an image.",
          );
          return;
        }

        video.srcObject = stream;
        await video.play().catch(() => undefined);
      } catch {
        if (!cancelled) {
          stopCamera();
          setIsWebcamOpen(false);
          setError(
            "Camera access was unavailable. You can still upload an image.",
          );
        }
      } finally {
        if (!cancelled) setIsStartingCamera(false);
      }
    }

    void initCamera();

    return () => {
      cancelled = true;
    };
  }, [isWebcamOpen, stopCamera]);

  const setPreviewFromFile = useCallback((file: File) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const nextPreview = URL.createObjectURL(file);
    previewUrlRef.current = nextPreview;
    setPreviewUrl(nextPreview);
  }, []);

  const clearFileInputs = useCallback(() => {
    if (uploadInputRef.current) uploadInputRef.current.value = "";
    if (cameraInputRef.current) cameraInputRef.current.value = "";
  }, []);

  const resetUpload = useCallback(() => {
    closeWebcam();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
    setFileName(null);
    setPrediction(null);
    setError(null);
    setIsDragging(false);
    clearFileInputs();
  }, [clearFileInputs, closeWebcam]);

  const openUploadPicker = useCallback(() => {
    if (isLoading || isWebcamOpen) return;
    uploadInputRef.current?.click();
  }, [isLoading, isWebcamOpen]);

  const openCameraPicker = useCallback(() => {
    if (isLoading || isWebcamOpen) return;

    // Phones/tablets: native rear-camera file capture where supported.
    // Desktop/laptop: in-page webcam when getUserMedia is available.
    if (shouldUseNativeCameraCapture()) {
      cameraInputRef.current?.click();
      return;
    }

    if (navigator.mediaDevices?.getUserMedia) {
      setError(null);
      setIsWebcamOpen(true);
      return;
    }

    // Unsupported desktop: fall back to file input (may open gallery).
    cameraInputRef.current?.click();
  }, [isLoading, isWebcamOpen]);

  const captureWebcamPhoto = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !isVideoReady) return;

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return;

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setError(
        "Camera access was unavailable. You can still upload an image.",
      );
      closeWebcam();
      return;
    }

    ctx.drawImage(video, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.9),
    );

    closeWebcam();

    if (!blob) {
      setError(
        "Camera access was unavailable. You can still upload an image.",
      );
      return;
    }

    const file = new File([blob], "captured-leaf.jpg", { type: "image/jpeg" });
    void handleSelectedFileRef.current(file);
  }, [isVideoReady, closeWebcam]);

  /** Shared path for gallery upload, mobile camera, webcam, and drag/drop. */
  const handleSelectedFile = useCallback(
    async (file: File) => {
      if (isLoading) return;

      if (!isAllowedImageFile(file)) {
        setError("Please upload a JPEG, PNG, WebP, or GIF image.");
        setPrediction(null);
        return;
      }

      if (file.size > MAX_UPLOAD_BYTES) {
        setError("File too large. Maximum upload size is 10MB.");
        setPrediction(null);
        return;
      }

      setFileName(file.name || "captured-leaf.jpg");
      setError(null);
      setPrediction(null);
      setIsLoading(true);
      setPreviewFromFile(file);

      const controller = new AbortController();
      const timeoutId = window.setTimeout(
        () => controller.abort(),
        REQUEST_TIMEOUT_MS,
      );

      try {
        const uploadFile = await resizeImageForUpload(file);
        setPreviewFromFile(uploadFile);

        const formData = new FormData();
        formData.append("file", uploadFile);

        const response = await fetch(PREDICT_URL, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        let payload: unknown = null;
        try {
          payload = await response.json();
        } catch {
          payload = null;
        }

        if (!response.ok) {
          const detail =
            payload &&
            typeof payload === "object" &&
            "detail" in payload &&
            typeof (payload as { detail: unknown }).detail === "string"
              ? (payload as { detail: string }).detail
              : response.status === 413
                ? "File too large. Maximum upload size is 10MB."
                : `Request failed (${response.status})`;
          throw new Error(detail);
        }

        if (!isPredictionPayload(payload)) {
          throw new Error("Unexpected response from the prediction server.");
        }

        setPrediction(payload);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setError("Request timed out, please try again.");
        } else if (err instanceof TypeError) {
          setError(
            "Could not reach the prediction server. Is it running on port 8000?",
          );
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Something went wrong while classifying the image.");
        }
        setPrediction(null);
      } finally {
        window.clearTimeout(timeoutId);
        setIsLoading(false);
        clearFileInputs();
      }
    },
    [isLoading, setPreviewFromFile, clearFileInputs],
  );

  // Keep a stable ref so captureWebcamPhoto can call the latest handler.
  handleSelectedFileRef.current = handleSelectedFile;

  const onFiles = useCallback(
    (files: FileList | null) => {
      if (isLoading) return;
      const file = files?.[0];
      if (file) void handleSelectedFile(file);
    },
    [isLoading, handleSelectedFile],
  );

  const topConfidencePct =
    prediction != null && typeof prediction.confidence === "number"
      ? Math.max(0, Math.min(100, prediction.confidence * 100))
      : 0;

  return (
    <div className="mx-auto flex w-full max-w-[46rem] flex-col gap-4">
      {/* Hidden inputs: gallery vs native mobile camera — both call handleSelectedFile */}
      <input
        ref={uploadInputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="sr-only"
        disabled={isLoading || isWebcamOpen}
        aria-hidden
        tabIndex={-1}
        onChange={(e) => {
          onFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        disabled={isLoading || isWebcamOpen}
        aria-hidden
        tabIndex={-1}
        onChange={(e) => {
          onFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {isWebcamOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-900/50 p-4"
          role="presentation"
          onClick={closeWebcam}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="webcam-dialog-title"
            className="w-full max-w-lg rounded-2xl border border-zinc-200 bg-white p-5 shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <h2
              id="webcam-dialog-title"
              className="text-lg font-semibold text-zinc-900"
            >
              Camera preview
            </h2>
            <p className="mt-1 text-sm text-zinc-500">
              Position a leaf in frame, then capture a photo.
            </p>

            <div className="relative mt-4 overflow-hidden rounded-xl border border-zinc-200 bg-zinc-900">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="aspect-video w-full object-cover"
                onLoadedMetadata={() => {
                  const video = videoRef.current;
                  if (video && video.videoWidth > 0 && video.videoHeight > 0) {
                    setIsVideoReady(true);
                  }
                }}
              />
              {(isStartingCamera || !isVideoReady) && (
                <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/70 text-sm text-white">
                  {isStartingCamera ? "Starting camera…" : "Preparing preview…"}
                </div>
              )}
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <button
                ref={captureButtonRef}
                type="button"
                disabled={!isVideoReady || isStartingCamera}
                onClick={() => void captureWebcamPhoto()}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <CameraIcon className="h-4 w-4" />
                Capture photo
              </button>
              <button
                type="button"
                onClick={closeWebcam}
                className="inline-flex min-h-11 items-center justify-center rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <section
        aria-label="Leaf image analyzer"
        onDragEnter={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!isLoading && !isWebcamOpen) setIsDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          e.stopPropagation();
          if (!isLoading && !isWebcamOpen) setIsDragging(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(false);
          if (!isLoading && !isWebcamOpen) onFiles(e.dataTransfer.files);
        }}
        className={`rounded-2xl border bg-white p-5 shadow-sm transition-colors sm:p-6 ${
          isDragging
            ? "border-emerald-500 bg-emerald-50/40"
            : "border-zinc-200"
        } ${isLoading ? "opacity-95" : ""}`}
      >
        {!previewUrl ? (
          <div className="flex flex-col items-center text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-emerald-50 ring-1 ring-emerald-100">
              <UploadIcon className="h-7 w-7 text-emerald-600" />
            </div>

            <h2 className="mt-4 text-lg font-semibold text-zinc-900">
              {isDragging
                ? "Drop image to analyze"
                : "Upload or capture a leaf photo"}
            </h2>
            <p className="mt-1.5 max-w-md text-sm leading-relaxed text-zinc-500">
              For best results, photograph one leaf clearly in good lighting.
            </p>

            <div className="mt-5 grid w-full grid-cols-1 gap-3 sm:grid-cols-2">
              <button
                type="button"
                disabled={isLoading || isWebcamOpen}
                onClick={openUploadPicker}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <UploadIcon className="h-4 w-4" />
                Upload image
              </button>
              <button
                type="button"
                disabled={isLoading || isWebcamOpen}
                onClick={openCameraPicker}
                className="inline-flex min-h-11 items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <CameraIcon className="h-4 w-4" />
                Take photo
              </button>
            </div>

            <p className="mt-4 text-sm text-zinc-500">
              or drag &amp; drop an image here
            </p>
            <p className="mt-3 text-xs text-zinc-400">
              JPEG, PNG, WebP or GIF · Max 10MB
            </p>
            <p className="mt-1 text-xs text-zinc-400">
              Images are resized before upload.
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="relative w-full max-w-lg">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt={
                  fileName
                    ? `Preview of leaf image ${fileName}`
                    : "Selected leaf image preview"
                }
                className="mx-auto max-h-[22rem] w-full rounded-xl border border-zinc-200 object-contain bg-zinc-50"
              />
              {isLoading && (
                <div
                  className="absolute inset-0 flex flex-col items-center justify-center gap-2 rounded-xl bg-white/75"
                  aria-hidden
                >
                  <span className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600" />
                  <span className="text-sm font-medium text-zinc-800">
                    Analyzing leaf…
                  </span>
                </div>
              )}
            </div>

            {!isLoading && (
              <div className="mt-4 flex w-full flex-col gap-2 sm:max-w-md sm:flex-row">
                <button
                  type="button"
                  onClick={openUploadPicker}
                  className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
                >
                  <UploadIcon className="h-4 w-4" />
                  Choose different image
                </button>
                <button
                  type="button"
                  onClick={openCameraPicker}
                  className="inline-flex min-h-11 flex-1 items-center justify-center gap-2 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
                >
                  <CameraIcon className="h-4 w-4" />
                  Take photo
                </button>
              </div>
            )}

            {fileName && !isLoading && (
              <p className="mt-2 max-w-full truncate px-2 text-xs text-zinc-400">
                {fileName}
              </p>
            )}
          </div>
        )}
      </section>

      <div className="rounded-xl border border-zinc-200 bg-white px-4 py-3.5 text-sm text-zinc-600 shadow-sm">
        <p className="font-medium text-zinc-800">For best results</p>
        <ul className="mt-2 space-y-1 text-zinc-500">
          <li>✓ Keep the leaf in focus</li>
          <li>✓ Use good lighting</li>
          <li>✓ Fill most of the frame with the leaf</li>
          <li>✓ Avoid extremely distant photos</li>
        </ul>
      </div>

      {isLoading && (
        <div
          className="flex items-center justify-center gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3.5 text-sm text-zinc-700 shadow-sm"
          role="status"
          aria-live="polite"
        >
          <span
            className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-300 border-t-emerald-600"
            aria-hidden
          />
          Analyzing leaf…
        </div>
      )}

      {error && !isLoading && (
        <div
          className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          role="alert"
        >
          {error}
        </div>
      )}

      {prediction && !isLoading && prediction.matched && prediction.class_name && (
        <div
          className="rounded-2xl border border-emerald-200 bg-emerald-50/80 px-5 py-5 shadow-sm"
          aria-live="polite"
        >
          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700">
            Likely match
          </p>
          <h2 className="mt-2 text-xl font-semibold leading-snug break-words text-zinc-900 sm:text-2xl">
            {formatClassName(prediction.class_name)}
          </h2>
          <p className="mt-2 inline-flex rounded-full bg-white px-3 py-1 text-sm font-medium text-emerald-800 ring-1 ring-emerald-200">
            {topConfidencePct.toFixed(1)}% confidence
          </p>

          <p className="mt-5 text-xs font-semibold uppercase tracking-wide text-emerald-800/80">
            Top matches
          </p>
          <div className="mt-3">
            <ConfidenceList
              items={prediction.predictions}
              accent="emerald"
              ariaLabel="Top matches"
            />
          </div>

          <button
            type="button"
            onClick={resetUpload}
            className="mt-5 inline-flex min-h-11 w-full items-center justify-center rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-medium text-white transition-colors hover:bg-emerald-800 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700"
          >
            Upload another image
          </button>
        </div>
      )}

      {prediction &&
        !isLoading &&
        !prediction.matched &&
        prediction.reason === "non_plant" && (
          <div
            className="rounded-2xl border border-slate-200 bg-slate-50 px-5 py-5 shadow-sm"
            aria-live="polite"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
              Not a supported leaf image
            </p>
            <p className="mt-2 text-base font-medium leading-relaxed text-zinc-900">
              {prediction.message ??
                "This image does not appear to contain a supported plant leaf."}
            </p>
            <p className="mt-2 text-sm text-zinc-600">
              FoliAI currently analyzes leaf images from bell pepper, potato, and
              tomato plants.
            </p>
            <button
              type="button"
              onClick={resetUpload}
              className="mt-5 inline-flex min-h-11 w-full items-center justify-center rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-500"
            >
              Try another image
            </button>
          </div>
        )}

      {prediction &&
        !isLoading &&
        !prediction.matched &&
        prediction.reason === "low_confidence" && (
          <div
            className="rounded-2xl border border-amber-200 bg-amber-50/80 px-5 py-5 shadow-sm"
            aria-live="polite"
          >
            <p className="text-xs font-semibold uppercase tracking-wide text-amber-800">
              No clear match
            </p>
            <p className="mt-2 text-base font-medium leading-relaxed text-zinc-900">
              {prediction.message ??
                "FoliAI detected a plant-like image, but couldn't confidently match it to one of the 15 supported classes."}
            </p>
            <p className="mt-2 text-sm text-zinc-600">
              FoliAI returns a match only when confidence reaches the configured
              threshold.
            </p>

            <p className="mt-5 text-xs font-semibold uppercase tracking-wide text-amber-800/80">
              Closest matches
            </p>
            <div className="mt-3">
              <ConfidenceList
                items={prediction.predictions}
                accent="amber"
                ariaLabel="Closest matches"
              />
            </div>

            <button
              type="button"
              onClick={resetUpload}
              className="mt-5 inline-flex min-h-11 w-full items-center justify-center rounded-xl border border-amber-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-amber-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber-600"
            >
              Upload another image
            </button>
          </div>
        )}

      {error && !isLoading && !prediction && (
        <button
          type="button"
          onClick={resetUpload}
          className="inline-flex min-h-11 w-full items-center justify-center rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:bg-zinc-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600"
        >
          Upload another image
        </button>
      )}

      <div className="space-y-1.5 text-center text-xs leading-relaxed text-zinc-400">
        <p>
          Currently supports 15 classes across bell pepper, potato, and tomato.
        </p>
        <p>
          Results can be less reliable for distant, blurry, unsupported, or
          heavily obstructed leaves.
        </p>
      </div>
    </div>
  );
}
