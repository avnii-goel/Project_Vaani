"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, Volume2, FileText, ArrowRight, X } from "lucide-react";

const LANGUAGES = [
  { code: "hi", native: "हिंदी", roman: "Hindi" },
  { code: "en", native: "English", roman: "English" },
  { code: "bn", native: "বাংলা", roman: "Bengali" },
  { code: "te", native: "తెలుగు", roman: "Telugu" },
  { code: "mr", native: "मराठी", roman: "Marathi" },
  { code: "ta", native: "தமிழ்", roman: "Tamil" },
] as const;

type LangCode = typeof LANGUAGES[number]["code"];

interface Turn {
  id: string;
  transcription: string;
  response: string;
  formFilename?: string;
}

// Custom hook for recording
function useRecorder(onDone: (blob: Blob) => void) {
  const [recording, setRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorder.current = mr;
      audioChunks.current = [];

      mr.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      mr.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: "audio/wav" });
        onDone(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mr.start();
      setRecording(true);
    } catch (e) {
      console.error("Microphone access error:", e);
      alert("Microphone access is required to use this application.");
    }
  }, [onDone]);

  const stop = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state === "recording") {
      mediaRecorder.current.stop();
      setRecording(false);
    }
  }, []);

  return { recording, start, stop };
}

function FormModal({ filename, onClose }: { filename: string; onClose: () => void }) {
  const [html, setHtml] = useState("<h2>Loading...</h2>");

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/form/${filename}`)
      .then(res => res.text())
      .then(setHtml)
      .catch(() => setHtml("<h2>Error loading form.</h2>"));
  }, [filename]);

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: "rgba(0,0,0,0.8)", zIndex: 50,
      display: "flex", alignItems: "center", justifyContent: "center", padding: "24px"
    }}>
      <div className="brutal-box" style={{ width: "100%", maxWidth: "800px", height: "85vh", display: "flex", flexDirection: "column" }}>
        <div style={{ backgroundColor: "#000000", color: "#ffffff", padding: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontFamily: "var(--font-mono)", margin: 0, textTransform: "uppercase" }}>{filename}</h2>
          <button onClick={onClose} style={{ background: "transparent", border: "none", cursor: "pointer", color: "#ffffff" }}>
            <X size={24} />
          </button>
        </div>
        <div style={{ flex: 1, backgroundColor: "#ffffff" }}>
          <iframe srcDoc={html} style={{ width: "100%", height: "100%", border: "none" }} />
        </div>
      </div>
    </div>
  );
}

function WelcomeScreen({ onSelect }: { onSelect: (c: LangCode) => void }) {
  const [active, setActive] = useState<LangCode | null>(null);

  return (
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "40px 24px", minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <header className="brutal-box" style={{ padding: "40px", marginBottom: "48px", position: "relative", overflow: "hidden" }}>
        <svg width="200" height="200" style={{ position: "absolute", top: -50, right: -50, zIndex: 0 }}>
          <circle cx="100" cy="100" r="100" fill="#fde047" />
        </svg>
        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={{ display: "inline-block", backgroundColor: "#000000", color: "#ffffff", padding: "8px 12px", fontFamily: "var(--font-mono)", fontWeight: "bold", textTransform: "uppercase", marginBottom: "24px" }}>
            Govt Welfare System
          </div>
          <h1 style={{ fontFamily: "var(--font-display)", fontSize: "5rem", margin: "0 0 16px 0", lineHeight: 1 }}>
            VAANI
          </h1>
          <p style={{ fontSize: "1.25rem", fontWeight: 500, maxWidth: "400px" }}>
            Speak in your native language. Discover the welfare schemes you deserve.
          </p>
        </div>
      </header>

      <div style={{ flex: 1 }}>
        <h2 style={{ fontFamily: "var(--font-mono)", fontSize: "1.5rem", textTransform: "uppercase", marginBottom: "24px", borderBottom: "4px solid #000000", paddingBottom: "8px", display: "inline-block" }}>
          Select Language
        </h2>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "24px" }}>
          {LANGUAGES.map((lang) => (
            <div
              key={lang.code}
              className={`lang-card ${active === lang.code ? "selected" : ""}`}
              onClick={() => setActive(lang.code)}
            >
              <div style={{ fontFamily: "var(--font-display)", fontSize: "2rem", color: active === lang.code ? "#ffffff" : "#000000" }}>
                {lang.native}
              </div>
              <div className="lang-subtext">
                {lang.roman}
              </div>
            </div>
          ))}
        </div>

        {active && (
          <button
            className="brutal-button"
            style={{ marginTop: "48px", width: "100%", fontSize: "1.5rem", padding: "24px" }}
            onClick={() => onSelect(active)}
          >
            START <ArrowRight size={32} />
          </button>
        )}
      </div>
    </div>
  );
}

function ConversationScreen({ lang, onReset }: { lang: LangCode; onReset: () => void }) {
  const langMeta = LANGUAGES.find((l) => l.code === lang)!;
  const [turns, setTurns] = useState<Turn[]>([]);
  const [processing, setProcessing] = useState(false);
  const [formModal, setFormModal] = useState<string | null>(null);
  const userId = useRef("vaani_" + Date.now());
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleAudio = useCallback(async (blob: Blob) => {
    setProcessing(true);
    try {
      const fd = new FormData();
      fd.append("audio", blob, "recording.wav");
      fd.append("userId", userId.current);
      fd.append("language", lang);

      const res = await fetch("/api/process-audio", { method: "POST", body: fd });
      const data = await res.json();

      if (data.success || data.spokenResponse) {
        setTurns((prev) => [...prev, {
          id: Date.now().toString(),
          transcription: data.transcription || "[No speech detected]",
          response: data.spokenResponse || data.response || "No response received.",
          formFilename: data.form_filename || undefined,
        }]);

        // Play the audio if ElevenLabs returned it!
        if (data.audioBase64) {
          try {
            const audioUrl = `data:audio/mpeg;base64,${data.audioBase64}`;
            const audio = new Audio(audioUrl);
            audio.play();
          } catch (e) {
            console.error("Failed to play audio:", e);
          }
        }
      } else {
        alert("Server error: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      console.error(e);
      alert("Failed to connect to the server.");
    } finally {
      setProcessing(false);
    }
  }, [lang]);

  const { recording, start, stop } = useRecorder(handleAudio);

  const toggle = () => {
    if (recording) stop();
    else start();
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, processing]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", maxWidth: "800px", margin: "0 auto", backgroundColor: "#ffffff", borderLeft: "4px solid #000000", borderRight: "4px solid #000000" }}>
      
      <header style={{ padding: "24px", borderBottom: "4px solid #000000", backgroundColor: "#fde047", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{ fontFamily: "var(--font-display)", fontSize: "2rem", fontWeight: "bold" }}>VAANI</span>
          <span style={{ fontFamily: "var(--font-mono)", fontWeight: "bold", border: "4px solid #000000", padding: "4px 12px", backgroundColor: "#ffffff" }}>
            {langMeta.native}
          </span>
        </div>
        <button onClick={onReset} className="brutal-box" style={{ padding: "8px 16px", fontFamily: "var(--font-mono)", fontWeight: "bold", cursor: "pointer" }}>
          CHANGE LANG
        </button>
      </header>

      <div style={{ flex: 1, overflowY: "auto", padding: "32px", display: "flex", flexDirection: "column", gap: "24px" }}>
        {turns.length === 0 && !processing && (
          <div className="brutal-box" style={{ padding: "40px", textAlign: "center", backgroundColor: "#f3f4f6" }}>
            <Volume2 size={64} style={{ margin: "0 auto 24px auto" }} />
            <h2 style={{ fontFamily: "var(--font-mono)", fontSize: "2rem", textTransform: "uppercase", marginBottom: "16px" }}>Ready to listen</h2>
            <p style={{ fontSize: "1.25rem", maxWidth: "400px", margin: "0 auto" }}>
              Press the big microphone button and speak in your language to find welfare schemes.
            </p>
          </div>
        )}

        {turns.map((turn) => (
          <div key={turn.id} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            <div className="chat-user">
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.875rem", textTransform: "uppercase", borderBottom: "4px solid #000000", paddingBottom: "4px", marginBottom: "8px", display: "inline-block" }}>YOU</div>
              <div>{turn.transcription}</div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "12px", alignSelf: "flex-start", maxWidth: "85%" }}>
              <div className="chat-ai">
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.875rem", textTransform: "uppercase", borderBottom: "4px solid #000000", paddingBottom: "4px", marginBottom: "8px", display: "inline-block" }}>VAANI</div>
                <div>{turn.response}</div>
              </div>

              {turn.formFilename && (
                <div className="brutal-box" style={{ padding: "16px", display: "flex", alignItems: "center", gap: "16px", backgroundColor: "#60a5fa", color: "#ffffff" }}>
                  <FileText size={32} />
                  <div>
                    <div style={{ fontFamily: "var(--font-mono)", fontWeight: "bold", fontSize: "1.25rem", textTransform: "uppercase" }}>Form Ready</div>
                  </div>
                  <button onClick={() => setFormModal(turn.formFilename!)} className="brutal-button" style={{ marginLeft: "auto", padding: "8px 16px", fontSize: "1rem" }}>
                    OPEN
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}

        {processing && (
          <div className="chat-ai" style={{ border: "4px dashed #000000" }}>
            <div style={{ fontFamily: "var(--font-mono)", fontWeight: "bold", fontSize: "1.25rem", textTransform: "uppercase" }}>PROCESSING...</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={{ padding: "32px", borderTop: "4px solid #000000", backgroundColor: "#f3f4f6", display: "flex", flexDirection: "column", alignItems: "center", gap: "24px" }}>
        <button
          onClick={toggle}
          disabled={processing}
          className={`mic-orb ${recording ? "active" : ""}`}
          style={{ opacity: processing ? 0.5 : 1 }}
        >
          {recording ? <Square size={40} fill="currentColor" /> : <Mic size={40} />}
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.25rem", fontWeight: "bold", textTransform: "uppercase", backgroundColor: "#ffffff", border: "4px solid #000000", padding: "8px 16px", boxShadow: "4px 4px 0px #000000" }}>
          {processing ? "PLEASE WAIT" : recording ? "RECORDING..." : "PRESS TO SPEAK"}
        </div>
      </div>

      {formModal && <FormModal filename={formModal} onClose={() => setFormModal(null)} />}
    </div>
  );
}

export default function Page() {
  const [lang, setLang] = useState<LangCode | null>(null);

  return (
    <>
      {!lang ? <WelcomeScreen onSelect={setLang} /> : <ConversationScreen lang={lang} onReset={() => setLang(null)} />}
    </>
  );
}
