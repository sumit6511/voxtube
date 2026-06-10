import { useState, useRef, useEffect } from "react";
import { Send, Loader2, ChevronDown } from "lucide-react";
import { api } from "../api";

interface Source {
  id: string;
  text: string;
  score: number;
}
interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  error?: boolean;
}

const SUGGESTIONS = [
  "What do viewers think about this video overall?",
  "What topics are most discussed in the comments?",
  "Are there any toxic or hateful comments?",
  "What do people say about the music / editing?",
  "Which aspects received the most praise?",
];

export default function ChatPanel({ jobId }: { jobId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send(question: string) {
    const q = question.trim();
    if (!q || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setLoading(true);

    try {
      const res = await api.chat(jobId, q);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: res.answer,
          sources: res.sources,
        },
      ]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Something went wrong.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: msg, error: true },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col gap-4">
      {/* ── Message history ────────────────────────────────────────── */}
      <div className="space-y-3 min-h-[100px]">
        {/* Suggestions — shown only when no messages yet */}
        {messages.length === 0 && (
          <div>
            <p className="label mb-3">Try asking</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  disabled={loading}
                  className="text-xs font-body text-gray-400 border border-base-border
                             hover:border-amber/40 hover:text-gray-200 px-3 py-2
                             rounded-lg transition-all text-left"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[88%] rounded-xl px-4 py-3 text-sm font-body leading-relaxed
              ${
                msg.role === "user"
                  ? "bg-amber/10 border border-amber/20 text-gray-200"
                  : msg.error
                    ? "bg-neg/5 border border-neg/20 text-neg"
                    : "card text-gray-300"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.text}</p>

              {msg.sources && msg.sources.length > 0 && (
                <SourcesCitation sources={msg.sources} />
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="flex justify-start">
            <div className="card px-4 py-3 flex items-center gap-2">
              <Loader2 size={14} className="animate-spin text-amber" />
              <span className="text-xs font-mono text-gray-500">Thinking…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input row ──────────────────────────────────────────────── */}
      <div className="flex gap-2">
        <input
          className="input-field flex-1 py-2.5 text-sm"
          placeholder="Ask about the comments…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send(input)}
          disabled={loading}
        />
        <button
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          className="btn-primary px-4 flex items-center gap-2 flex-shrink-0"
        >
          {loading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Send size={15} />
          )}
        </button>
      </div>

      <p className="text-xs font-mono text-gray-700">
        Powered by Llama 3.2 via Ollama · answers grounded in retrieved comments  
      </p>
    </div>
  );
}

// ── Source citations collapse ──────────────────────────────────────────────

function SourcesCitation({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="mt-3 border-t border-base-border pt-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 text-xs font-mono text-gray-600
                   hover:text-gray-400 transition-colors"
      >
        <ChevronDown
          size={12}
          className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
        {sources.length} source{sources.length !== 1 ? "s" : ""} used
      </button>

      {open && (
        <div className="mt-2 space-y-1.5">
          {sources.map((s) => (
            <div
              key={s.id}
              className="text-xs text-gray-500 font-body bg-base rounded-lg px-3 py-2
                         border border-base-border/50"
            >
              <span className="font-mono text-amber/60 mr-2">
                {(s.score * 100).toFixed(0)}%
              </span>
              {s.text.length > 120 ? `${s.text.slice(0, 120)}…` : s.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
