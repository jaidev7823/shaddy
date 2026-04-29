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

function floatTo16BitPCM(input: Float32Array) {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    // Clamp values between -1 and 1
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return output;
}

// Add this helper function outside your component
async function startRecording() {
  setError(null);
  setStatus("Initializing...");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Force 16kHz to match your backend VAD/Whisper models
    const audioContext = new AudioContext({ sampleRate: 16000 }); 
    const source = audioContext.createMediaStreamSource(stream);
    
    // 4096 samples at 16kHz is ~250ms of audio per chunk
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    const ws = new WebSocket(`${WS_URL}/ws/audio`);
    webSocketRef.current = ws;

    ws.onopen = () => {
      setStatus("Connected & Streaming");
      console.log("WebSocket opened successfully");
    };

    processor.onaudioprocess = (e) => {
      if (ws.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        const pcm16 = floatTo16BitPCM(inputData);
        
        // Convert Int16Array to Base64
        const uint8View = new Uint8Array(pcm16.buffer);
        let binary = "";
        for (let i = 0; i < uint8View.byteLength; i++) {
          binary += String.fromCharCode(uint8View[i]);
        }
        const base64Audio = btoa(binary);

        ws.send(JSON.stringify({
          type: "audio_chunk",
          data: {
            audio: base64Audio,
            sample_rate: 16000
          }
        }));
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    // Keep references for cleanup
    (window as any).audioStream = stream;
    (window as any).audioContext = audioContext;
    
    setIsRecording(true);
    setStatus("Recording...");

  } catch (err: any) {
    console.error("Recording error:", err);
    setError(err.message || "Could not start recording");
    setStatus("Error");
  }
} // <--- This was likely the missing brace causing the semicolon error

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
