"use client";

import { useRef, useState } from "react";

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  onStatusChange: (recording: boolean) => void;
  disabled?: boolean;
}

export default function AudioRecorder({ onRecordingComplete, onStatusChange, disabled }: AudioRecorderProps) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        onRecordingComplete(blob);
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      setRecording(true);
      onStatusChange(true);
    } catch (err) {
      console.error("Mic error:", err);
    }
  };

  const stop = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    onStatusChange(false);
  };

  return { recording, start, stop, toggle: recording ? stop : start };
}
