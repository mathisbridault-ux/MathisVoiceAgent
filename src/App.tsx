/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from "@google/genai";
import { Mic, MicOff, Volume2, VolumeX, Loader2, Sparkles, User, Briefcase, GraduationCap, Languages } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { floatTo16BitPCM, base64ToArrayBuffer, arrayBufferToBase64 } from './lib/audio-utils';

// CV Data for System Instruction
const MATHIS_CV = `
Name: Mathis Bridault
Role: Communication & Video Creation Specialist
Languages: French (C2), English (B2)
Education:
- Master of Science Communication du Luxe & de la Mode (Sup de Pub, Paris, 2025-2027)
- Master 1 Abroad Brand Strategy (Sup de Pub, London, 2024)
- Bachelor Communication Globale (Sup de Pub, Paris, 2023-2024)
- Bachelor Innovation Numérique (Web School Factory, Paris, 2021-2023)
Experience:
- Assistant Communication at Prodigious (Publicis Groupe, 2025): Internal/External communication, data analysis, social media, newsletter.
- Freelance Video Creation (2025-Present): Scriptwriting, creative ideas, filming, editing.
- Assistant Communication at Events&nous (2024): Communication strategy, marketing campaigns, festival management.
- Vendeur at Intermarché (2024)
- Magasinier at Decathlon (2022)
Skills: Consumer studies, Visual identity design, Design thinking, Storytelling, Digital strategy.
Soft Skills: Creativity, Analytical mind, Adaptability, Collaboration, Curiosity.
Hobbies: Travel, Photography, Cooking, Golf.
`;

const SYSTEM_INSTRUCTION = `
You are Jackie, a voice assistant for Mathis Bridault.
You speak both English and French.
You have a British accent when speaking English.
Your personality is professional, helpful, and slightly witty, like a high-end concierge.
You know everything about Mathis's professional background, education, and skills based on his CV.
Mathis is currently looking for an apprenticeship starting in September 2025.

Key Guidelines:
1. Always start the conversation by saying: "Hello I'm Jackie, what would you like to learn about Mathis?" (or the French equivalent if appropriate).
2. If asked about Mathis, use the provided CV data to answer accurately.
3. You can switch between English and French seamlessly depending on the user's language.
4. Keep responses concise and suitable for voice interaction.

CV Data:
${MATHIS_CV}
`;

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [volume, setVolume] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<AudioWorkletNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const isPlayingRef = useRef(false);
  const nextStartTimeRef = useRef<number>(0);

  const setupAudio = async () => {
    if (!audioContextRef.current) {
      // Gemini Live API usually outputs at 24kHz
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      await audioContextRef.current.audioWorklet.addModule('data:text/javascript;base64,' + btoa(`
        class AudioProcessor extends AudioWorkletProcessor {
          process(inputs, outputs, parameters) {
            const input = inputs[0];
            if (input.length > 0) {
              const channelData = input[0];
              this.port.postMessage(channelData);
            }
            return true;
          }
        }
        registerProcessor('audio-processor', AudioProcessor);
      `));
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  };

  const playQueuedAudio = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0 || !audioContextRef.current) return;

    isPlayingRef.current = true;
    setIsSpeaking(true);

    // Initial start time
    if (nextStartTimeRef.current < audioContextRef.current.currentTime) {
      nextStartTimeRef.current = audioContextRef.current.currentTime + 0.05;
    }

    while (audioQueueRef.current.length > 0) {
      const chunk = audioQueueRef.current.shift()!;
      const buffer = audioContextRef.current.createBuffer(1, chunk.length, 24000);
      buffer.getChannelData(0).set(chunk);

      const source = audioContextRef.current.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContextRef.current.destination);

      const analyzer = audioContextRef.current.createAnalyser();
      source.connect(analyzer);
      analyzer.fftSize = 256;
      const dataArray = new Uint8Array(analyzer.frequencyBinCount);

      const updateVolume = () => {
        if (!isPlayingRef.current) return;
        analyzer.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        setVolume(average / 128);
        requestAnimationFrame(updateVolume);
      };
      updateVolume();

      source.start(nextStartTimeRef.current);
      nextStartTimeRef.current += buffer.duration;

      // Wait for the buffer to finish playing before moving to the next one
      // but keep scheduling ahead to avoid gaps
      const waitTime = (nextStartTimeRef.current - audioContextRef.current.currentTime) * 1000;
      if (waitTime > 100) {
        await new Promise(resolve => setTimeout(resolve, waitTime - 50));
      }
    }

    isPlayingRef.current = false;
    setIsSpeaking(false);
    setVolume(0);
  }, []);

  const connect = async () => {
    try {
      setIsConnecting(true);
      setError(null);
      
      // Request mic access immediately after user interaction
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      await setupAudio();

      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      
      const session = await ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-12-2025",
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
        },
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setIsConnecting(false);
            initializeMicProcessor();
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn?.parts) {
              for (const part of message.serverContent.modelTurn.parts) {
                if (part.inlineData?.data) {
                  const audioData = base64ToArrayBuffer(part.inlineData.data);
                  const pcmData = new Int16Array(audioData);
                  const float32Data = new Float32Array(pcmData.length);
                  for (let i = 0; i < pcmData.length; i++) {
                    float32Data[i] = pcmData[i] / 32768.0;
                  }
                  audioQueueRef.current.push(float32Data);
                  playQueuedAudio();
                }
              }
            }
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              isPlayingRef.current = false;
              setIsSpeaking(false);
              nextStartTimeRef.current = 0;
            }
          },
          onclose: () => {
            setIsConnected(false);
            stopMic();
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError("Connection error. Please try again.");
            setIsConnecting(false);
          }
        }
      });

      sessionRef.current = session;
    } catch (err) {
      console.error("Connection failed:", err);
      if (err instanceof Error && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
        setError("Microphone access was denied. Please check your browser settings.");
      } else {
        setError("Failed to connect to Jackie.");
      }
      setIsConnecting(false);
      stopMic();
    }
  };

  const initializeMicProcessor = () => {
    try {
      if (audioContextRef.current && streamRef.current) {
        const source = audioContextRef.current.createMediaStreamSource(streamRef.current);
        const processor = new AudioWorkletNode(audioContextRef.current, 'audio-processor');

        processor.port.onmessage = (event) => {
          if (isMuted) return;
          const pcmBuffer = floatTo16BitPCM(event.data);
          const base64Data = arrayBufferToBase64(pcmBuffer);
          sessionRef.current?.sendRealtimeInput({
            audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
          });
        };

        source.connect(processor);
        processorRef.current = processor;
      }
    } catch (err) {
      console.error("Mic processor initialization failed:", err);
      setError("Failed to initialize microphone.");
    }
  };

  const stopMic = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    processorRef.current?.disconnect();
    processorRef.current = null;
  };

  const toggleMute = () => setIsMuted(!isMuted);

  const disconnect = () => {
    sessionRef.current?.close();
    setIsConnected(false);
    stopMic();
  };

  return (
    <div className="min-h-screen bg-[#0a0502] text-[#f5f2ed] font-sans selection:bg-[#ff4e00]/30 overflow-hidden flex flex-col">
      {/* Background Atmosphere */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[60%] h-[60%] bg-[#3a1510] rounded-full blur-[120px] opacity-40 animate-pulse" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-[#ff4e00] rounded-full blur-[150px] opacity-20" />
      </div>

      {/* Header */}
      <header className="relative z-10 p-8 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-[#ff4e00] flex items-center justify-center shadow-[0_0_20px_rgba(255,78,0,0.4)]">
            <Sparkles size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Jackie</h1>
            <p className="text-xs text-[#f5f2ed]/50 uppercase tracking-widest">Mathis's Assistant</p>
          </div>
        </div>
        
        {isConnected && (
          <button 
            onClick={disconnect}
            className="px-4 py-2 rounded-full border border-[#f5f2ed]/20 text-xs uppercase tracking-widest hover:bg-[#f5f2ed]/10 transition-colors"
          >
            End Session
          </button>
        )}
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center p-6">
        <AnimatePresence mode="wait">
          {!isConnected ? (
            <motion.div 
              key="landing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center max-w-2xl"
            >
              <h2 className="text-5xl md:text-7xl font-light mb-8 leading-tight">
                Meet <span className="italic font-serif">Jackie</span>.
              </h2>
              <p className="text-lg text-[#f5f2ed]/70 mb-12 max-w-md mx-auto leading-relaxed">
                Your voice-first guide to Mathis Bridault's professional journey, skills, and aspirations.
              </p>
              
              <button
                onClick={connect}
                disabled={isConnecting}
                className="group relative px-12 py-5 bg-white text-black rounded-full font-medium text-lg overflow-hidden transition-transform active:scale-95 disabled:opacity-50"
              >
                <div className="absolute inset-0 bg-[#ff4e00] translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                <span className="relative z-10 group-hover:text-white transition-colors flex items-center gap-3">
                  {isConnecting ? (
                    <>
                      <Loader2 className="animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    <>
                      Start Conversation
                      <Volume2 size={20} />
                    </>
                  )}
                </span>
              </button>
              
              {error && (
                <p className="mt-6 text-[#ff4e00] text-sm font-medium">{error}</p>
              )}
            </motion.div>
          ) : (
            <motion.div 
              key="active"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="w-full max-w-4xl flex flex-col items-center"
            >
              {/* Voice Visualizer */}
              <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center mb-16">
                {/* Outer Rings */}
                <motion.div 
                  animate={{ 
                    scale: isSpeaking ? [1, 1.1, 1] : 1,
                    opacity: isSpeaking ? [0.2, 0.4, 0.2] : 0.1
                  }}
                  transition={{ repeat: Infinity, duration: 2 }}
                  className="absolute inset-0 border border-[#ff4e00] rounded-full"
                />
                <motion.div 
                  animate={{ 
                    scale: isSpeaking ? [1, 1.2, 1] : 1,
                    opacity: isSpeaking ? [0.1, 0.2, 0.1] : 0.05
                  }}
                  transition={{ repeat: Infinity, duration: 3, delay: 0.5 }}
                  className="absolute inset-[-20%] border border-[#ff4e00] rounded-full"
                />

                {/* Core Visualizer */}
                <div className="relative z-10 w-full h-full flex items-center justify-center gap-1">
                  {[...Array(12)].map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{ 
                        height: isSpeaking ? `${20 + (volume * 80 * Math.random())}%` : '8px',
                        opacity: isSpeaking ? 0.8 : 0.3
                      }}
                      className="w-2 bg-[#ff4e00] rounded-full"
                    />
                  ))}
                </div>
              </div>

              {/* Status & Controls */}
              <div className="text-center mb-12">
                <h3 className="text-2xl font-light tracking-wide mb-2">
                  {isSpeaking ? "Jackie is speaking..." : "Jackie is listening..."}
                </h3>
                <p className="text-[#f5f2ed]/40 text-sm uppercase tracking-[0.3em]">
                  Live Session Active
                </p>
              </div>

              <div className="flex gap-6">
                <button
                  onClick={toggleMute}
                  className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                    isMuted ? 'bg-[#ff4e00] text-white' : 'bg-[#f5f2ed]/10 text-[#f5f2ed] hover:bg-[#f5f2ed]/20'
                  }`}
                >
                  {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
                </button>
                <div className="w-16 h-16 rounded-full bg-[#f5f2ed]/5 flex items-center justify-center text-[#f5f2ed]/30">
                  {isSpeaking ? <Volume2 size={24} /> : <VolumeX size={24} />}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer / Info Cards */}
      <footer className="relative z-10 p-8 grid grid-cols-1 md:grid-cols-4 gap-4 max-w-7xl mx-auto w-full">
        <div className="p-4 rounded-2xl bg-[#f5f2ed]/5 border border-[#f5f2ed]/10 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2 text-[#ff4e00]">
            <GraduationCap size={16} />
            <span className="text-[10px] uppercase tracking-widest font-bold">Education</span>
          </div>
          <p className="text-xs text-[#f5f2ed]/70">MSc Communication Luxe & Mode @ Sup de Pub</p>
        </div>
        <div className="p-4 rounded-2xl bg-[#f5f2ed]/5 border border-[#f5f2ed]/10 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2 text-[#ff4e00]">
            <Briefcase size={16} />
            <span className="text-[10px] uppercase tracking-widest font-bold">Experience</span>
          </div>
          <p className="text-xs text-[#f5f2ed]/70">Publicis Prodigious, Freelance Video, Events&nous</p>
        </div>
        <div className="p-4 rounded-2xl bg-[#f5f2ed]/5 border border-[#f5f2ed]/10 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2 text-[#ff4e00]">
            <Languages size={16} />
            <span className="text-[10px] uppercase tracking-widest font-bold">Languages</span>
          </div>
          <p className="text-xs text-[#f5f2ed]/70">French (Native), English (British Accent Jackie)</p>
        </div>
        <div className="p-4 rounded-2xl bg-[#f5f2ed]/5 border border-[#f5f2ed]/10 backdrop-blur-sm">
          <div className="flex items-center gap-2 mb-2 text-[#ff4e00]">
            <User size={16} />
            <span className="text-[10px] uppercase tracking-widest font-bold">Contact</span>
          </div>
          <p className="text-xs text-[#f5f2ed]/70">mathis.bridault@gmail.com | Paris, FR</p>
        </div>
      </footer>
    </div>
  );
}
