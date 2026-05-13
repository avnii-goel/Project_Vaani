"use client";

import { useEffect, useState } from "react";
import { X, Download, FileText } from "lucide-react";

interface FormModalProps {
  formFilename: string;
  onClose: () => void;
}

export default function FormModal({ formFilename, onClose }: FormModalProps) {
  const [html, setHtml] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchForm = async () => {
      try {
        setLoading(true);
        const res = await fetch(`http://127.0.0.1:8000/form/${formFilename}`);
        if (!res.ok) throw new Error(`Status ${res.status}`);
        setHtml(await res.text());
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    fetchForm();
  }, [formFilename]);

  const download = () => {
    const blob = new Blob([html], { type: "text/html" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: formFilename || "form.html",
    });
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.75)", backdropFilter: "blur(8px)" }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="glass rounded-2xl w-full max-w-3xl h-[88vh] flex flex-col overflow-hidden"
        style={{ border: "1px solid rgba(124,108,240,0.25)", boxShadow: "0 0 60px rgba(124,108,240,0.15)" }}>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: "rgba(255,255,255,0.08)" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{ background: "rgba(6,182,212,0.15)", border: "1px solid rgba(6,182,212,0.3)" }}>
              <FileText size={18} style={{ color: "#06b6d4" }} />
            </div>
            <div>
              <p className="font-semibold text-white" style={{ fontFamily: "var(--font-space-grotesk)" }}>Application Form</p>
              <p className="text-xs" style={{ color: "rgba(240,238,255,0.5)" }}>{formFilename}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!loading && !error && (
              <button onClick={download}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all"
                style={{ background: "rgba(6,182,212,0.12)", border: "1px solid rgba(6,182,212,0.25)", color: "#06b6d4" }}>
                <Download size={15} /> Download
              </button>
            )}
            <button onClick={onClose}
              className="w-9 h-9 rounded-xl flex items-center justify-center transition-all hover:bg-white/10"
              style={{ border: "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.6)" }}>
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-hidden">
          {loading && (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <div className="flex gap-1.5">
                {[0, 1, 2].map((i) => (
                  <div key={i} className="dot" style={{ animationDelay: `${i * 0.2}s` }} />
                ))}
              </div>
              <p style={{ color: "rgba(240,238,255,0.5)", fontSize: "14px" }}>Loading form…</p>
            </div>
          )}
          {error && (
            <div className="flex flex-col items-center justify-center h-full gap-3">
              <p style={{ color: "#f87171" }}>Could not load form</p>
              <p style={{ color: "rgba(240,238,255,0.4)", fontSize: "13px" }}>{error}</p>
            </div>
          )}
          {!loading && !error && (
            <iframe
              srcDoc={html}
              title="Application Form"
              className="w-full h-full"
              sandbox="allow-forms allow-scripts"
            />
          )}
        </div>
      </div>
    </div>
  );
}
