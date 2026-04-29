"use client";

import { useEffect, useState, useRef } from "react";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

interface AudioMessage {
  type: "audio_chunk" | "ping" | "close";
  data?: {
    audio: string;
    sample_rate: number;
  };
}

interface ServerMessage {
  type: "status" | "response" | "error";
  data?: Record<string, any>;
}

export default function Recorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [response, setResponse] = useState<ServerMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string>("");
  const [llmResponse, setLLMResponse] = useState<any>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const webSocketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      if (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN) {
        webSocketRef.current.send(JSON.stringify({ type: "close" }));
        webSocketRef.current.close();
      }
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
    };
  }, [isRecording]);

  async function startRecording() {
    setError(null);
    setTranscript("");
    setLLMResponse(null);
    setStatus("Initializing...");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000 },
      });
      setStatus("Connected to microphone");

      const ws = new WebSocket(`${WS_URL}/ws/audio`);

      ws.onopen = () => {
        setStatus("Connected to backend");
      };

      ws.onmessage = (event) => {
        try {
          const msg: ServerMessage = JSON.parse(event.data);
          setResponse(msg);

          if (msg.type === "status") {
            const state = msg.data?.state || msg.data?.message || "";
            setStatus(state);
            if (msg.data?.text) {
              setTranscript(msg.data.text);
            }
          } else if (msg.type === "response") {
            if (msg.data?.transcript) {
              setTranscript(msg.data.transcript);
            }
            if (msg.data?.llm_response) {
              setLLMResponse(msg.data.llm_response);
            }
            setStatus("Nudge generated");
          } else if (msg.type === "error") {
            setError(msg.data?.message || "Backend error");
            setStatus("Error");
          }
        } catch (err) {
          console.error("Failed to parse message:", err);
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
        setStatus("Disconnected");
      };

      ws.onclose = () => {
        setStatus("Disconnected");
        if (isRecording) {
          mediaRecorderRef.current?.stop();
          setIsRecording(false);
        }
      };

      webSocketRef.current = ws;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
          const arrayBuffer = await event.data.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);
          const base64 = btoa(String.fromCharCode.apply(null, Array.from(uint8Array)));

          const message: AudioMessage = {
            type: "audio_chunk",
            data: {
              audio: base64,
              sample_rate: 16000,
            },
          };

          ws.send(JSON.stringify(message));
        }
      };

      mediaRecorder.start(100);
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setStatus("Recording... Click to stop");
    } catch (err: any) {
      setError(err.message || "Failed to access microphone");
      setStatus("Error");
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    if (webSocketRef.current && webSocketRef.current.readyState === WebSocket.OPEN) {
      webSocketRef.current.send(JSON.stringify({ type: "close" }));
      webSocketRef.current.close();
    }
    setIsRecording(false);
    setStatus("Stopped");
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 px-6 py-8">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-8">
        <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
          <div className="mb-6 flex flex-col gap-2">
            <h1 className="text-4xl font-semibold text-white">🎙️ Shady Real-Time Audio</h1>
            <p className="max-w-2xl text-slate-400">
              Click the button below to start recording. Audio is streamed to the backend in real-time.
            </p>
          </div>

          <div className="flex flex-wrap gap-4 items-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`px-8 py-4 rounded-2xl font-semibold text-lg transition ${
                isRecording
                  ? "bg-red-500 hover:bg-red-600 text-white"
                  : "bg-teal-500 hover:bg-teal-600 text-slate-950"
              }`}
            >
              {isRecording ? "🛑 Stop Recording" : "🎤 Start Recording"}
            </button>

            <div className="flex-1 rounded-2xl bg-slate-800 px-4 py-3 text-sm font-medium text-slate-300">
              {status}
            </div>
          </div>
        </section>

        {error ? (
          <div className="rounded-2xl border border-rose-500 bg-rose-500/10 px-6 py-4 text-sm text-rose-300">
            <span className="font-semibold">Error:</span> {error}
          </div>
        ) : null}

        {transcript ? (
          <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
            <h2 className="text-xl font-semibold text-white mb-4">📝 Transcript</h2>
            <div className="rounded-2xl border border-slate-700 bg-slate-950 p-6 text-slate-200 whitespace-pre-wrap break-words">
              {transcript}
            </div>
          </section>
        ) : null}

        {llmResponse ? (
          <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
            <h2 className="text-xl font-semibold text-white mb-4">💡 LLM Feedback</h2>
            <div className="grid gap-4">
              {llmResponse.should_nudge ? (
                <div className="rounded-2xl border border-teal-500 bg-teal-500/10 p-6">
                  <div className="text-sm font-semibold text-teal-300 mb-2">Nudge</div>
                  <p className="text-teal-100 mb-4 text-lg">{llmResponse.nudge}</p>
                  {llmResponse.why ? (
                    <div className="text-sm text-teal-200 italic">
                      Why: {llmResponse.why}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="rounded-2xl border border-slate-700 bg-slate-900 p-6">
                  <div className="text-sm text-slate-400">No nudge needed for this input.</div>
                </div>
              )}

              {llmResponse.lesson_id ? (
                <div className="rounded-2xl border border-slate-700 bg-slate-800 p-4">
                  <div className="text-xs text-slate-400">Lesson ID</div>
                  <div className="text-sm font-mono text-slate-200">{llmResponse.lesson_id}</div>
                </div>
              ) : null}
            </div>
          </section>
        ) : null}

        {response ? (
          <section className="rounded-3xl border border-slate-700 bg-slate-900/90 p-8 shadow-xl shadow-slate-950/30">
            <h2 className="text-xl font-semibold text-white mb-4">📡 Raw Response</h2>
            <pre className="rounded-2xl border border-slate-700 bg-slate-950 p-6 overflow-x-auto text-xs text-slate-300">
              {JSON.stringify(response, null, 2)}
            </pre>
          </section>
        ) : null}
      </div>
    </main>
  );
}
