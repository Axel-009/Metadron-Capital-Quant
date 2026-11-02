import { useState, useEffect, useRef, useCallback } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

interface JarvisMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  model?: string;
  voice_mode?: boolean;
}

interface JarvisStatus {
  agent_id: string;
  status: "online" | "degraded" | "offline";
  model: string;
  llama_available: boolean;
  llm_bridge_available: boolean;
  tts_available: boolean;
  stt_available: boolean;
  request_count: number;
  error_count: number;
  pending_recommendations: number;
  timestamp: string;
}

interface Recommendation {
  id: number;
  content: string;
  action: string;
  prefix: string;
  timestamp: string;
  dismissed: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Status Bar
// ═══════════════════════════════════════════════════════════════════════════

function JarvisStatusBar({ status }: { status: JarvisStatus | null }) {
  if (!status) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 border-b border-terminal-border/50 text-[10px] font-mono">
        <span className="w-1.5 h-1.5 rounded-full bg-terminal-negative animate-pulse" />
        <span className="text-terminal-text-faint">JARVIS — CONNECTING...</span>
      </div>
    );
  }

  const statusColor =
    status.status === "online"
      ? "bg-terminal-positive"
      : status.status === "degraded"
      ? "bg-terminal-warning"
      : "bg-terminal-negative";

  const statusLabel =
    status.status === "online"
      ? "ONLINE"
      : status.status === "degraded"
      ? "DEGRADED"
      : "OFFLINE";

  return (
    <div className="flex items-center gap-4 px-4 py-2 border-b border-terminal-border/50 text-[10px] font-mono flex-shrink-0">
      <div className="flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
        <span className="text-terminal-text-primary font-semibold tracking-widest">OPEN JARVIS</span>
        <span className="text-terminal-text-faint">— {statusLabel}</span>
      </div>

      <div className="h-3 w-px bg-terminal-border/50" />

      <div className="flex items-center gap-1">
        <span className="text-terminal-text-faint">MODEL</span>
        <span
          className={
            status.llama_available ? "text-terminal-positive" : "text-terminal-warning"
          }
        >
          {status.llama_available ? "LLAMA 3.1-8B (LOCAL)" : "LLM BRIDGE (FALLBACK)"}
        </span>
      </div>

      <div className="h-3 w-px bg-terminal-border/50" />

      <div className="flex items-center gap-3">
        <span className={status.tts_available ? "text-terminal-positive" : "text-terminal-text-faint"}>
          {status.tts_available ? "● TTS" : "○ TTS"}
        </span>
        <span className={status.stt_available ? "text-terminal-positive" : "text-terminal-text-faint"}>
          {status.stt_available ? "● STT" : "○ STT"}
        </span>
      </div>

      <div className="h-3 w-px bg-terminal-border/50" />

      <span className="text-terminal-text-faint">
        {status.request_count} REQ / {status.error_count} ERR
      </span>

      {status.pending_recommendations > 0 && (
        <>
          <div className="h-3 w-px bg-terminal-border/50" />
          <span className="text-terminal-warning animate-pulse">
            {status.pending_recommendations} PENDING REC
          </span>
        </>
      )}

      <span className="ml-auto text-terminal-text-faint opacity-50">
        INSTRUCTION-ONLY · READ + RECOMMEND
      </span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Message Bubble
// ═══════════════════════════════════════════════════════════════════════════

function MessageBubble({ msg }: { msg: JarvisMessage }) {
  const isUser = msg.role === "user";
  const time = new Date(msg.timestamp).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} mb-4`}>
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-sm flex items-center justify-center flex-shrink-0 text-[9px] font-bold tracking-wider ${
          isUser
            ? "bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/30"
            : "bg-purple-500/20 text-purple-300 border border-purple-500/30"
        }`}
      >
        {isUser ? "AJ" : "J"}
      </div>

      {/* Bubble */}
      <div className={`max-w-[78%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        <div
          className={`px-3 py-2.5 rounded text-[11px] font-mono leading-relaxed whitespace-pre-wrap ${
            isUser
              ? "bg-terminal-accent/10 border border-terminal-accent/20 text-terminal-text-primary"
              : "bg-white/[0.04] border border-terminal-border/50 text-terminal-text-primary"
          }`}
        >
          {msg.content}
        </div>
        <div className="flex items-center gap-2 text-[9px] text-terminal-text-faint">
          <span>{time}</span>
          {!isUser && msg.model && (
            <span className="text-purple-400/60">{msg.model.toUpperCase()}</span>
          )}
          {!isUser && msg.voice_mode && (
            <span className="text-terminal-text-faint/40">VOICE</span>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Recommendations Panel
// ═══════════════════════════════════════════════════════════════════════════

function RecommendationsPanel({
  recs,
  onDismiss,
}: {
  recs: Recommendation[];
  onDismiss: (id: number) => void;
}) {
  if (recs.length === 0) return null;

  return (
    <div className="border-t border-terminal-warning/20 bg-terminal-warning/5 px-4 py-3 flex-shrink-0">
      <div className="text-[9px] font-mono tracking-widest text-terminal-warning mb-2">
        PENDING RECOMMENDATIONS — REQUIRES NANOCLAW APPROVAL
      </div>
      <div className="flex flex-col gap-2">
        {recs.map((rec) => (
          <div
            key={rec.id}
            className="flex items-start gap-3 bg-terminal-warning/5 border border-terminal-warning/20 rounded px-3 py-2"
          >
            <div className="flex-1 text-[10px] font-mono text-terminal-text-primary leading-relaxed">
              {rec.content}
            </div>
            <button
              onClick={() => onDismiss(rec.id)}
              className="text-[9px] text-terminal-text-faint hover:text-terminal-negative transition-colors flex-shrink-0"
            >
              DISMISS
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Voice Button
// ═══════════════════════════════════════════════════════════════════════════

function VoiceButton({
  onTranscript,
  sttAvailable,
  disabled,
}: {
  onTranscript: (text: string) => void;
  sttAvailable: boolean;
  disabled: boolean;
}) {
  const [recording, setRecording] = useState(false);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    if (!sttAvailable) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => chunksRef.current.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        try {
          const res = await fetch("/api/engine/jarvis/voice", {
            method: "POST",
            body: formData,
          });
          if (res.ok) {
            const data = await res.json();
            if (data.transcript) onTranscript(data.transcript);
          }
        } catch (err) {
          console.error("Voice upload error:", err);
        }

        stream.getTracks().forEach((t) => t.stop());
      };

      recorder.start();
      mediaRef.current = recorder;
      setRecording(true);
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }, [sttAvailable, onTranscript]);

  const stopRecording = useCallback(() => {
    if (mediaRef.current && recording) {
      mediaRef.current.stop();
      mediaRef.current = null;
      setRecording(false);
    }
  }, [recording]);

  return (
    <button
      onMouseDown={startRecording}
      onMouseUp={stopRecording}
      onTouchStart={startRecording}
      onTouchEnd={stopRecording}
      disabled={disabled || !sttAvailable}
      title={
        !sttAvailable
          ? "STT unavailable — install openjarvis[speech]"
          : "Hold to speak"
      }
      className={`w-8 h-8 rounded flex items-center justify-center transition-all flex-shrink-0 ${
        recording
          ? "bg-terminal-negative/20 border border-terminal-negative text-terminal-negative animate-pulse"
          : sttAvailable
          ? "bg-white/[0.04] border border-terminal-border hover:border-purple-400 text-terminal-text-muted hover:text-purple-300"
          : "bg-white/[0.02] border border-terminal-border/30 text-terminal-text-faint/30 cursor-not-allowed"
      }`}
    >
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <rect x="5" y="1" width="4" height="7" rx="2" stroke="currentColor" strokeWidth="1.2" />
        <path d="M2.5 7.5C2.5 10.09 4.515 12 7 12s4.5-1.91 4.5-4.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        <line x1="7" y1="12" x2="7" y2="13.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    </button>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Jarvis Chat Panel
// ═══════════════════════════════════════════════════════════════════════════

function JarvisChatPanel() {
  const [messages, setMessages] = useState<JarvisMessage[]>([]);
  const [status, setStatus] = useState<JarvisStatus | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // ── Fetch initial state ──────────────────────────────────────────────────

  useEffect(() => {
    fetch("/api/engine/jarvis/messages")
      .then((r) => r.json())
      .then((d) => { if (d.messages) setMessages(d.messages); })
      .catch(() => {});

    fetch("/api/engine/jarvis/status")
      .then((r) => r.json())
      .then((d) => setStatus(d))
      .catch(() => {});

    fetch("/api/engine/jarvis/recommendations")
      .then((r) => r.json())
      .then((d) => { if (d.recommendations) setRecommendations(d.recommendations); })
      .catch(() => {});
  }, []);

  // ── Poll status + recommendations ────────────────────────────────────────

  useEffect(() => {
    const iv = setInterval(() => {
      fetch("/api/engine/jarvis/status")
        .then((r) => r.json())
        .then((d) => setStatus(d))
        .catch(() => {});

      fetch("/api/engine/jarvis/recommendations")
        .then((r) => r.json())
        .then((d) => { if (d.recommendations) setRecommendations(d.recommendations); })
        .catch(() => {});
    }, 15000);
    return () => clearInterval(iv);
  }, []);

  // ── Auto scroll ──────────────────────────────────────────────────────────

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Send message ─────────────────────────────────────────────────────────

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || streaming) return;
      const userMsg = text.trim();
      setInput("");

      const userEntry: JarvisMessage = {
        role: "user",
        content: userMsg,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userEntry]);
      setStreaming(true);

      // Optimistic streaming entry
      const assistantEntry: JarvisMessage = {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        model: status?.llama_available ? "llama-3.1-8b" : "llm-bridge",
        voice_mode: voiceMode,
      };
      setMessages((prev) => [...prev, assistantEntry]);

      try {
        const res = await fetch("/api/engine/jarvis/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMsg, voice_mode: voiceMode }),
        });

        if (!res.ok || !res.body) throw new Error("Stream failed");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const raw = decoder.decode(value);
          const lines = raw.split("\n");
          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const chunk = line.slice(6);
              if (chunk === "[DONE]" || chunk.startsWith("[ERROR]")) break;
              fullText += chunk;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: fullText,
                };
                return updated;
              });
            }
          }
        }

        // Auto-play TTS if voice mode enabled and TTS available
        if (voiceMode && status?.tts_available && fullText) {
          try {
            const ttsRes = await fetch("/api/engine/jarvis/speak", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: fullText }),
            });
            if (ttsRes.ok) {
              const audioBlob = await ttsRes.blob();
              const audioUrl = URL.createObjectURL(audioBlob);
              const audio = new Audio(audioUrl);
              audio.play().catch(() => {});
            }
          } catch {
            // TTS playback failed silently
          }
        }
      } catch (err) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: "[Jarvis bridge unavailable — check engine API connection]",
          };
          return updated;
        });
      } finally {
        setStreaming(false);
        inputRef.current?.focus();
      }
    },
    [streaming, voiceMode, status]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(input);
      }
    },
    [input, sendMessage]
  );

  const handleDismiss = useCallback(async (id: number) => {
    try {
      await fetch(`/api/engine/jarvis/recommendations/${id}/dismiss`, {
        method: "POST",
      });
      setRecommendations((prev) => prev.filter((r) => r.id !== id));
    } catch {}
  }, []);

  const handleClearHistory = useCallback(async () => {
    try {
      await fetch("/api/engine/jarvis/history", { method: "DELETE" });
      setMessages([]);
    } catch {}
  }, []);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full">
      {/* Status bar */}
      <JarvisStatusBar status={status} />

      {/* Recommendations */}
      <RecommendationsPanel recs={recommendations} onDismiss={handleDismiss} />

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 min-h-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            {/* Jarvis icon */}
            <div className="w-16 h-16 rounded-full bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="#a855f7" strokeWidth="1.5" strokeOpacity="0.6" />
                <circle cx="16" cy="16" r="8" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.4" />
                <circle cx="16" cy="16" r="3" fill="#a855f7" fillOpacity="0.8" />
                <line x1="16" y1="2" x2="16" y2="8" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.4" />
                <line x1="16" y1="24" x2="16" y2="30" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.4" />
                <line x1="2" y1="16" x2="8" y2="16" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.4" />
                <line x1="24" y1="16" x2="30" y2="16" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.4" />
              </svg>
            </div>
            <div>
              <p className="text-[13px] font-mono text-terminal-text-primary tracking-wider">
                OPEN JARVIS
              </p>
              <p className="text-[10px] text-terminal-text-faint mt-1">
                Backed by Llama 3.1-8B · Voice + Text · Instruction-only
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2 max-w-md w-full">
              {[
                "What's the current portfolio status?",
                "Summarise today's market conditions",
                "What is the cube regime right now?",
                "Give me a live system health check",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-left px-3 py-2 text-[10px] font-mono text-terminal-text-muted hover:text-terminal-text-primary bg-white/[0.02] hover:bg-white/[0.04] border border-terminal-border/50 hover:border-terminal-border rounded transition-all"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} msg={msg} />
            ))}
            {streaming && (
              <div className="flex items-center gap-2 pl-10 mb-4">
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </div>
                <span className="text-[9px] text-terminal-text-faint font-mono">
                  JARVIS THINKING...
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Input area */}
      <div className="border-t border-terminal-border/50 px-4 py-3 flex-shrink-0 bg-terminal-surface/50">
        <div className="flex items-end gap-2">
          {/* Voice toggle */}
          <button
            onClick={() => setVoiceMode((v) => !v)}
            title={voiceMode ? "Voice mode ON — responses optimised for TTS" : "Voice mode OFF"}
            className={`w-8 h-8 rounded flex items-center justify-center flex-shrink-0 transition-all ${
              voiceMode
                ? "bg-purple-500/20 border border-purple-500/50 text-purple-300"
                : "bg-white/[0.04] border border-terminal-border text-terminal-text-muted hover:text-terminal-text-primary"
            }`}
          >
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
              <path d="M6.5 1.5C7.88 1.5 9 2.62 9 4v3c0 1.38-1.12 2.5-2.5 2.5S4 8.38 4 7V4c0-1.38 1.12-2.5 2.5-2.5z" stroke="currentColor" strokeWidth="1.1" />
              <path d="M2 6.5C2 9.26 4 11.5 6.5 11.5S11 9.26 11 6.5" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
              <line x1="6.5" y1="11.5" x2="6.5" y2="13" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
            </svg>
          </button>

          {/* STT record button */}
          <VoiceButton
            onTranscript={(t) => { setInput(t); inputRef.current?.focus(); }}
            sttAvailable={status?.stt_available ?? false}
            disabled={streaming}
          />

          {/* Text input */}
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              voiceMode
                ? "Speak or type — voice mode active..."
                : "Ask Jarvis anything... (Enter to send, Shift+Enter for newline)"
            }
            rows={1}
            disabled={streaming}
            className="flex-1 bg-transparent border border-terminal-border/60 rounded px-3 py-2 text-[11px] font-mono text-terminal-text-primary placeholder:text-terminal-text-faint/40 focus:outline-none focus:border-purple-400/50 resize-none min-h-[34px] max-h-[120px] overflow-y-auto disabled:opacity-50"
            style={{ lineHeight: "1.5" }}
          />

          {/* Send */}
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || streaming}
            className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0 bg-purple-500/15 border border-purple-500/30 text-purple-300 hover:bg-purple-500/25 hover:border-purple-400 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
              <path d="M2 6.5H11M7.5 2.5L11.5 6.5L7.5 10.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          {/* Clear history */}
          {messages.length > 0 && (
            <button
              onClick={handleClearHistory}
              disabled={streaming}
              title="Clear conversation history"
              className="w-8 h-8 rounded flex items-center justify-center flex-shrink-0 bg-white/[0.02] border border-terminal-border/40 text-terminal-text-faint hover:text-terminal-negative hover:border-terminal-negative/40 transition-all disabled:opacity-30"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 3h8M5 3V2h2v1M3 3l.5 7h5L9 3" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center justify-between mt-1.5">
          <span className="text-[9px] text-terminal-text-faint/40 font-mono">
            {voiceMode ? "VOICE MODE · TTS AUTO-PLAY" : "TEXT MODE · ENTER TO SEND"}
          </span>
          <span className="text-[9px] text-terminal-text-faint/30 font-mono">
            INSTRUCTION-ONLY · WRITES BLOCKED · LLAMA 3.1-8B
          </span>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Page
// ═══════════════════════════════════════════════════════════════════════════

export default function OpenJarvisPage() {
  return (
    <DashboardPanel title="OPEN JARVIS" className="h-full">
      <JarvisChatPanel />
    </DashboardPanel>
  );
}
