import { NextRequest, NextResponse } from "next/server";
import { Groq } from "groq-sdk";
import { createReadStream, writeFileSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY! });
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000/chat";
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || "";

const ELEVENLABS_CONFIG: Record<string, { voiceId: string; modelId: string }> = {
  hi: { voiceId: 'pNInz6obpgDQGcFmaJgB', modelId: 'eleven_multilingual_v2' },
  en: { voiceId: '21m00Tcm4TlvDq8ikWAM', modelId: 'eleven_monolingual_v1' },
  bn: { voiceId: 'pNInz6obpgDQGcFmaJgB', modelId: 'eleven_multilingual_v2' },
  te: { voiceId: 'pNInz6obpgDQGcFmaJgB', modelId: 'eleven_multilingual_v2' },
  mr: { voiceId: 'pNInz6obpgDQGcFmaJgB', modelId: 'eleven_multilingual_v2' },
  ta: { voiceId: 'pNInz6obpgDQGcFmaJgB', modelId: 'eleven_multilingual_v2' }
};

const LANG_NAMES: Record<string, string> = {
  hi: "Hindi", bn: "Bengali", te: "Telugu", mr: "Marathi", ta: "Tamil"
};

export async function POST(req: NextRequest) {
  let tempPath = "";
  try {
    const fd = await req.formData();
    const audio = fd.get("audio") as File | null;
    const userId = (fd.get("userId") as string) || "vaani_user";
    const lang = (fd.get("language") as string) || "en";

    if (!audio) {
      return NextResponse.json({ error: "No audio file" }, { status: 400 });
    }

    /* 1 — Groq Whisper: transcribe & translate to English */
    const buf = Buffer.from(await audio.arrayBuffer());
    tempPath = join(tmpdir(), `vaani-${Date.now()}.wav`);
    writeFileSync(tempPath, buf);

    const transcription = await groq.audio.translations.create({
      file: createReadStream(tempPath),
      model: "whisper-large-v3",
    });
    const englishText = transcription.text;
    console.log(`[whisper] -> "${englishText}"`);

    /* 2 — Call Python FastAPI backend */
    let responseText = "";
    let formFilename: string | null = null;

    try {
      const apiRes = await fetch(BACKEND_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ message: englishText, user_id: userId }),
        signal: AbortSignal.timeout(90_000),
      });

      if (!apiRes.ok) throw new Error(`Backend ${apiRes.status}`);
      const data = await apiRes.json();
      responseText = (data.response || "").split("(")[0].trim();
      formFilename = data.form_filename || null;
      console.log(`[backend] -> "${responseText}" | form: ${formFilename}`);
    } catch (err) {
      console.error("[backend error]", err);
      responseText = "I could not reach the backend right now. Please try again.";
    }

    /* 3 — Translate response back to Target Language via Groq */
    let finalResponseText = responseText;
    if (lang !== "en" && responseText) {
      const targetLang = LANG_NAMES[lang] || lang;
      console.log(`[translate] Translating to ${targetLang}...`);

      const chatCompletion = await groq.chat.completions.create({
        messages: [
          {
            role: 'system',
            content: `You are a precise ${targetLang} translator. Your ONLY job is to translate the EXACT English text provided into natural ${targetLang}. DO NOT add ANY introduction, greeting, or extra information. DO NOT change or expand the original meaning. Simply translate EXACTLY what is provided.`
          },
          { role: 'user', content: `Please translate this English text to ${targetLang}: "${responseText}"` },
        ],
        model: 'meta-llama/llama-4-scout-17b-16e-instruct',
        temperature: 0.1,
      });
      finalResponseText = chatCompletion.choices[0]?.message?.content || responseText;
      console.log(`[translate] Translated: "${finalResponseText}"`);
    }

    /* 4 — ElevenLabs Text-to-Speech */
    let audioBase64 = null;
    try {
      const ttsConfig = ELEVENLABS_CONFIG[lang];
      if (!ttsConfig) throw new Error(`No TTS config for lang: ${lang}`);

      console.log(`[tts] Generating audio with ElevenLabs for voice: ${ttsConfig.voiceId}`);

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg'
      };

      if (ELEVENLABS_API_KEY) {
        headers['xi-api-key'] = ELEVENLABS_API_KEY;
      }

      const ttsResponse = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${ttsConfig.voiceId}`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          text: finalResponseText,
          model_id: ttsConfig.modelId,
          voice_settings: { stability: 0.5, similarity_boost: 0.5 }
        })
      });

      if (!ttsResponse.ok) {
        const errText = await ttsResponse.text();
        console.error("[tts error] ElevenLabs failed:", errText);
      } else {
        const audioBuffer = await ttsResponse.arrayBuffer();
        audioBase64 = Buffer.from(audioBuffer).toString('base64');
        console.log(`[tts] Successfully generated audio!`);
      }
    } catch (ttsErr) {
      console.error("[tts error] Failed to process TTS:", ttsErr);
    }

    return NextResponse.json({
      transcription: englishText,
      spokenResponse: finalResponseText,
      form_filename: formFilename,
      audioBase64: audioBase64,
      success: true,
    });

  } catch (err: any) {
    console.error("[process-audio]", err);
    return NextResponse.json({ error: err.message }, { status: 500 });
  } finally {
    if (tempPath) { try { unlinkSync(tempPath); } catch { } }
  }
}
