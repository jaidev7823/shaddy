"use client";

import { useMemo, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("Ready");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<any>(null);

  const backendUrl = useMemo(() => BACKEND_URL, []);

  async function handleHealthCheck() {
    setError(null);
    setStatus("Checking backend...");
    try {
      const response = await fetch(`${backendUrl}/health`);
      const body = await response.json();
      setHealth(body);
      setStatus("Backend healthy");
    } catch (err) {
      setError("Failed to reach backend. Is it running?");
      setStatus("Backend unavailable");
    }
  }

  async function handleUpload(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please choose an audio file to upload.");
      return;
    }

    setLoading(true);
    setStatus("Uploading audio...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${backendUrl}/process-audio`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Upload failed");
      }

      const body = await response.json();
      setResult(body);
      setStatus("Audio processed successfully");
    } catch (err: any) {
      setError(err.message || "Unexpected error");
      setStatus("Processing failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 px-6 py-8">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
        <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
          <div className="mb-6 flex flex-col gap-2">
            <h1 className="text-4xl font-semibold text-white">Shady Audio Debug UI</h1>
            <p className="max-w-2xl text-slate-400">
              Upload a WAV or audio file to the backend and inspect transcription, speaker
              verification, and LLM feedback.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-[1fr_auto]">
            <button
              type="button"
              onClick={handleHealthCheck}
              className="rounded-2xl bg-slate-700 px-5 py-3 text-sm font-medium text-slate-100 transition hover:bg-slate-600"
            >
              Check Backend Health
            </button>
            <div className="rounded-2xl bg-slate-800 px-5 py-3 text-sm text-slate-400">
              {health ? (
                <pre className="whitespace-pre-wrap text-slate-200">
                  {JSON.stringify(health, null, 2)}
                </pre>
              ) : (
                <span>Backend URL: {backendUrl}</span>
              )}
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
          <form onSubmit={handleUpload} className="flex flex-col gap-5">
            <div className="flex flex-col gap-2">
              <label className="text-sm font-semibold text-slate-200" htmlFor="audio-file">
                Upload audio file
              </label>
              <input
                id="audio-file"
                type="file"
                accept="audio/*"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none transition focus:border-slate-500"
              />
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="submit"
                disabled={loading}
                className="rounded-2xl bg-teal-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-teal-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Processing…" : "Upload and Process"}
              </button>
              <span className="inline-flex items-center rounded-2xl bg-slate-800 px-4 py-3 text-sm text-slate-400">
                {status}
              </span>
            </div>
          </form>

          {error ? (
            <div className="rounded-2xl border border-rose-500 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
              {error}
            </div>
          ) : null}

          {result ? (
            <div className="mt-8 grid gap-4 rounded-3xl border border-slate-700 bg-slate-950/80 p-6">
              <div>
                <h2 className="text-xl font-semibold text-white">Result</h2>
                <p className="mt-2 text-slate-400">
                  Review the backend response for transcription and nudging.
                </p>
              </div>

              <div className="grid gap-3 rounded-2xl border border-slate-800 bg-slate-900 p-4">
                <div className="grid gap-1 text-sm text-slate-300">
                  <span className="font-medium text-slate-100">Transcript</span>
                  <pre className="whitespace-pre-wrap text-slate-100">{result.transcript}</pre>
                </div>
                <div className="grid gap-1 text-sm text-slate-300 md:grid-cols-2">
                  <div>
                    <span className="font-medium text-slate-100">Speaker is student</span>
                    <p>{result.speaker_is_student ? "Yes" : "No"}</p>
                  </div>
                  <div>
                    <span className="font-medium text-slate-100">Similarity</span>
                    <p>{result.speaker_similarity?.toFixed(3) ?? "N/A"}</p>
                  </div>
                </div>
                <div className="grid gap-1 text-sm text-slate-300">
                  <span className="font-medium text-slate-100">LLM Response</span>
                  <pre className="whitespace-pre-wrap text-slate-100">{JSON.stringify(result.llm_response, null, 2)}</pre>
                </div>
                <div className="text-sm text-slate-300">
                  <span className="font-medium text-slate-100">Audio generated</span>
                  <p>{result.audio_generated ? "Yes" : "No"}</p>
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
