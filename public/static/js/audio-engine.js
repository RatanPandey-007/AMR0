/**
 * AMR Predict — Immersive Sound Design & Interactive Web Audio System
 * Inspired by the cinematic, atmospheric audio architecture of georgedo.com
 *
 * Full-Spectrum Features:
 *  1. Generative Cinematic Ambient Electronic Soundtrack:
 *     - 100% self-contained Web Audio synthesis (zero external audio file dependencies).
 *     - Multi-layer generative drone, evolving scientific A-minor/D-minor harmonic pads,
 *       crystalline fluorophore shimmer chimes, and a 30-BPM biophotonic resting pulse.
 *     - Smooth 2.5s fade-ins and 1.2s fade-outs; infinitely non-repetitive; seamless looping.
 *  2. Interaction Sound Effects Engine:
 *     - Navigation & Chapters: Soft futuristic clicks, pentatonic chapter transition chords.
 *     - Buttons & CTAs: Crisp haptic clicks, deep resonant confirmation pulses for primary actions.
 *     - 3D Viewport: Confocal, Dark-Field, and Cryo-EM tonal feedback, structural layer filter blips.
 *     - Forms & Intelligence: Droplet selection clicks, validation error warnings, prediction success triads.
 *     - Automatic event delegation across all buttons, dropdowns, and navigation elements.
 *  3. Header Audio Control Panel & Flyout Settings:
 *     - Independent Master Audio, Ambient Soundtrack, and Interaction SFX switches.
 *     - Continuous 0-100% volume slider with real-time feedback.
 *     - Animated 4-bar equalizer wave indicator reflecting live soundtrack synthesis.
 *     - Persistent memory via localStorage (amr_audio_master, amr_audio_music, amr_audio_sfx, amr_audio_volume).
 *  4. First-Visit "ENTER WITH SOUND" Welcome Dialog:
 *     - Strictly complies with modern browser autoplay policies.
 *     - Elegant floating card offering "Enter with Sound" vs "Continue Silently".
 *  5. Window Lifecycle & Concurrency Protection:
 *     - Dynamic master limiter (DynamicsCompressorNode) prevents clipping or distortion.
 *     - Auto-fades out when browser tab is hidden (visibilitychange), restores on return.
 *     - Polyphony limiter and debouncing for rapid clicks.
 */

(function () {
    'use strict';

    // ─── State Management & Local Storage Defaults ─────────────────────────────
    const STORAGE_KEY_MASTER = 'amr_audio_master';
    const STORAGE_KEY_MUSIC = 'amr_audio_music';
    const STORAGE_KEY_SFX = 'amr_audio_sfx';
    const STORAGE_KEY_VOLUME = 'amr_audio_volume';
    const STORAGE_KEY_DISMISSED = 'amr_audio_prompt_dismissed';

    let state = {
        masterEnabled: localStorage.getItem(STORAGE_KEY_MASTER) !== 'false',
        musicEnabled: localStorage.getItem(STORAGE_KEY_MUSIC) !== 'false',
        sfxEnabled: localStorage.getItem(STORAGE_KEY_SFX) !== 'false',
        volume: parseFloat(localStorage.getItem(STORAGE_KEY_VOLUME) || '0.6'),
        promptDismissed: localStorage.getItem(STORAGE_KEY_DISMISSED) === 'true',
        isInitialized: false,
        isSoundtrackPlaying: false,
        isTabVisible: !document.hidden
    };

    // ─── Web Audio Graph Nodes ──────────────────────────────────────────────────
    let audioCtx = null;
    let masterGain = null;
    let musicBus = null;
    let sfxBus = null;
    let compressor = null;

    // Soundtrack Node References
    let soundtrackNodes = {
        droneOsc1: null,
        droneOsc2: null,
        droneSub: null,
        droneFilter: null,
        droneLfo: null,
        droneLfoGain: null,
        padVoices: [],
        padFilter: null,
        pulseTimer: null,
        shimmerTimer: null,
        chordTimer: null
    };

    // SFX Rate Limiting & Concurrency State
    let lastSfxTime = 0;
    let lastSfxType = '';
    let activeVoiceCount = 0;
    const MAX_POLYPHONY = 6;

    // ─── 1. Initialize Audio Graph ──────────────────────────────────────────────
    function initAudioContext() {
        if (state.isInitialized && audioCtx) {
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            return true;
        }

        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) {
            console.warn('AMR Audio: Web Audio API is not supported in this browser.');
            return false;
        }

        try {
            audioCtx = new AudioContextClass();

            // Master Limiter / Dynamics Compressor to prevent clipping
            compressor = audioCtx.createDynamicsCompressor();
            compressor.threshold.setValueAtTime(-6, audioCtx.currentTime);
            compressor.knee.setValueAtTime(12, audioCtx.currentTime);
            compressor.ratio.setValueAtTime(4, audioCtx.currentTime);
            compressor.attack.setValueAtTime(0.003, audioCtx.currentTime);
            compressor.release.setValueAtTime(0.25, audioCtx.currentTime);
            compressor.connect(audioCtx.destination);

            // Master Gain Node
            masterGain = audioCtx.createGain();
            const initGain = state.masterEnabled ? state.volume : 0.0001;
            masterGain.gain.setValueAtTime(initGain, audioCtx.currentTime);
            masterGain.connect(compressor);

            // Music Sub-Bus
            musicBus = audioCtx.createGain();
            musicBus.gain.setValueAtTime(state.musicEnabled ? 0.35 : 0.0001, audioCtx.currentTime);
            musicBus.connect(masterGain);

            // SFX Sub-Bus
            sfxBus = audioCtx.createGain();
            sfxBus.gain.setValueAtTime(state.sfxEnabled ? 0.65 : 0.0001, audioCtx.currentTime);
            sfxBus.connect(masterGain);

            state.isInitialized = true;
            return true;
        } catch (err) {
            console.error('AMR Audio Initialization Failed:', err);
            return false;
        }
    }

    // ─── 2. Generative Ambient Electronic Soundtrack ───────────────────────────
    function startSoundtrack() {
        if (!initAudioContext()) return;
        if (state.isSoundtrackPlaying) return;

        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const now = audioCtx.currentTime;

        // ── A. Deep Sub-Harmonic Foundation Drone (55Hz / 27.5Hz with LFO Filter) ─
        const droneFilter = audioCtx.createBiquadFilter();
        droneFilter.type = 'lowpass';
        droneFilter.frequency.setValueAtTime(180, now);
        droneFilter.Q.setValueAtTime(2.8, now);
        droneFilter.connect(musicBus);
        soundtrackNodes.droneFilter = droneFilter;

        // LFO for slow breathing filter sweep (0.04 Hz = 25s cycle)
        const lfo = audioCtx.createOscillator();
        lfo.type = 'sine';
        lfo.frequency.setValueAtTime(0.04, now);
        const lfoGain = audioCtx.createGain();
        lfoGain.gain.setValueAtTime(65, now);
        lfo.connect(lfoGain);
        lfoGain.connect(droneFilter.frequency);
        lfo.start(now);
        soundtrackNodes.droneLfo = lfo;
        soundtrackNodes.droneLfoGain = lfoGain;

        // Fundamental A1 (55Hz) Sine
        const osc1 = audioCtx.createOscillator();
        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(55, now);
        const osc1Gain = audioCtx.createGain();
        osc1Gain.gain.setValueAtTime(0.38, now);
        osc1.connect(osc1Gain);
        osc1Gain.connect(droneFilter);
        osc1.start(now);
        soundtrackNodes.droneOsc1 = osc1;

        // Detuned Harmonic A1+ (55.4Hz) Triangle for analog warmth
        const osc2 = audioCtx.createOscillator();
        osc2.type = 'triangle';
        osc2.frequency.setValueAtTime(55.4, now);
        const osc2Gain = audioCtx.createGain();
        osc2Gain.gain.setValueAtTime(0.18, now);
        osc2.connect(osc2Gain);
        osc2Gain.connect(droneFilter);
        osc2.start(now);
        soundtrackNodes.droneOsc2 = osc2;

        // Sub-Bass A0 (27.5Hz) Pure Sine
        const sub = audioCtx.createOscillator();
        sub.type = 'sine';
        sub.frequency.setValueAtTime(27.5, now);
        const subGain = audioCtx.createGain();
        subGain.gain.setValueAtTime(0.24, now);
        sub.connect(subGain);
        subGain.connect(droneFilter);
        sub.start(now);
        soundtrackNodes.droneSub = sub;

        // ── B. Evolving Scientific Harmonic Pad (A minor 9th / D minor 9th) ─────
        const padFilter = audioCtx.createBiquadFilter();
        padFilter.type = 'bandpass';
        padFilter.frequency.setValueAtTime(440, now);
        padFilter.Q.setValueAtTime(1.8, now);
        padFilter.connect(musicBus);
        soundtrackNodes.padFilter = padFilter;

        // Scientific Harmonic Chords (Frequencies in Hz)
        const chordProgression = [
            [110.0, 164.81, 246.94, 261.63, 392.00], // Am9: A2, E3, B3, C4, G4
            [146.83, 174.61, 220.00, 261.63, 329.63], // Dm9: D3, F3, A3, C4, E4
            [130.81, 164.81, 196.00, 246.94, 329.63], // Cmaj7: C3, E3, G3, B3, E4
            [87.31, 130.81, 174.61, 220.00, 261.63]   // Fmaj7: F2, C3, F3, A3, C4
        ];
        let chordIndex = 0;

        function setPadChord(chord) {
            soundtrackNodes.padVoices.forEach(v => {
                try {
                    v.gain.gain.linearRampToValueAtTime(0.0001, audioCtx.currentTime + 1.8);
                    setTimeout(() => v.osc.stop(), 2000);
                } catch (e) {}
            });
            soundtrackNodes.padVoices = [];

            const cTime = audioCtx.currentTime;
            chord.forEach((freq, idx) => {
                const osc = audioCtx.createOscillator();
                osc.type = idx % 2 === 0 ? 'sine' : 'triangle';
                osc.frequency.setValueAtTime(freq + (Math.random() - 0.5) * 0.4, cTime);

                const voiceGain = audioCtx.createGain();
                voiceGain.gain.setValueAtTime(0.0001, cTime);
                voiceGain.gain.linearRampToValueAtTime(0.08 / chord.length, cTime + 2.2);

                osc.connect(voiceGain);
                voiceGain.connect(padFilter);
                osc.start(cTime);
                soundtrackNodes.padVoices.push({ osc, gain: voiceGain });
            });
        }

        setPadChord(chordProgression[0]);

        // Evolve chord every 12 seconds
        soundtrackNodes.chordTimer = setInterval(() => {
            if (!state.isSoundtrackPlaying || !audioCtx) return;
            chordIndex = (chordIndex + 1) % chordProgression.length;
            setPadChord(chordProgression[chordIndex]);
        }, 12000);

        // ── C. Crystalline Biophotonic Shimmer Chimes (Generative Pentatonic) ───
        const shimmerPitches = [523.25, 659.25, 783.99, 1046.50, 1318.51, 1567.98]; // C5, E5, G5, C6, E6, G6

        function triggerShimmerChime() {
            if (!state.isSoundtrackPlaying || !audioCtx) return;
            if (state.musicEnabled && state.masterEnabled) {
                const sTime = audioCtx.currentTime;
                const freq = shimmerPitches[Math.floor(Math.random() * shimmerPitches.length)];

                const sOsc = audioCtx.createOscillator();
                sOsc.type = 'sine';
                sOsc.frequency.setValueAtTime(freq, sTime);

                const sGain = audioCtx.createGain();
                sGain.gain.setValueAtTime(0.0001, sTime);
                sGain.gain.linearRampToValueAtTime(0.028, sTime + 0.12);
                sGain.gain.exponentialRampToValueAtTime(0.0001, sTime + 1.8);

                let panner = null;
                if (audioCtx.createStereoPanner) {
                    panner = audioCtx.createStereoPanner();
                    panner.pan.setValueAtTime((Math.random() - 0.5) * 1.6, sTime);
                    sOsc.connect(sGain);
                    sGain.connect(panner);
                    panner.connect(musicBus);
                } else {
                    sOsc.connect(sGain);
                    sGain.connect(musicBus);
                }

                sOsc.start(sTime);
                sOsc.stop(sTime + 1.9);
            }

            // Schedule next shimmer at random interval between 3.5s and 7.5s
            const nextDelay = 3500 + Math.random() * 4000;
            soundtrackNodes.shimmerTimer = setTimeout(triggerShimmerChime, nextDelay);
        }

        soundtrackNodes.shimmerTimer = setTimeout(triggerShimmerChime, 4000);

        // ── D. 30-BPM Biophotonic Resting Pulse (Subtle Incubator Rhythm) ───────
        function triggerRestingPulse() {
            if (!state.isSoundtrackPlaying || !audioCtx) return;
            if (state.musicEnabled && state.masterEnabled) {
                const pTime = audioCtx.currentTime;
                const pOsc = audioCtx.createOscillator();
                pOsc.type = 'sine';
                pOsc.frequency.setValueAtTime(42, pTime);
                pOsc.frequency.exponentialRampToValueAtTime(32, pTime + 0.35);

                const pGain = audioCtx.createGain();
                pGain.gain.setValueAtTime(0.0001, pTime);
                pGain.gain.linearRampToValueAtTime(0.05, pTime + 0.04);
                pGain.gain.exponentialRampToValueAtTime(0.0001, pTime + 0.38);

                pOsc.connect(pGain);
                pGain.connect(musicBus);

                pOsc.start(pTime);
                pOsc.stop(pTime + 0.4);
            }
            soundtrackNodes.pulseTimer = setTimeout(triggerRestingPulse, 2000); // 30 BPM
        }

        soundtrackNodes.pulseTimer = setTimeout(triggerRestingPulse, 1800);

        // Smooth 2.5s Fade-In
        musicBus.gain.cancelScheduledValues(now);
        musicBus.gain.setValueAtTime(0.0001, now);
        musicBus.gain.linearRampToValueAtTime(state.musicEnabled ? 0.35 : 0.0001, now + 2.5);

        state.isSoundtrackPlaying = true;
        updateUI();
    }

    function stopSoundtrack() {
        if (!state.isSoundtrackPlaying || !audioCtx) return;

        const now = audioCtx.currentTime;
        if (musicBus) {
            musicBus.gain.cancelScheduledValues(now);
            musicBus.gain.linearRampToValueAtTime(0.0001, now + 1.2);
        }

        clearInterval(soundtrackNodes.chordTimer);
        clearTimeout(soundtrackNodes.shimmerTimer);
        clearTimeout(soundtrackNodes.pulseTimer);

        setTimeout(() => {
            try {
                if (soundtrackNodes.droneOsc1) soundtrackNodes.droneOsc1.stop();
                if (soundtrackNodes.droneOsc2) soundtrackNodes.droneOsc2.stop();
                if (soundtrackNodes.droneSub) soundtrackNodes.droneSub.stop();
                if (soundtrackNodes.droneLfo) soundtrackNodes.droneLfo.stop();
                soundtrackNodes.padVoices.forEach(v => { try { v.osc.stop(); } catch(e){} });
                soundtrackNodes.padVoices = [];
            } catch (e) {}
            state.isSoundtrackPlaying = false;
            updateUI();
        }, 1300);
    }

    // ─── 3. Full-Spectrum Reactive Sound Effects Engine ─────────────────────────
    function playSfx(type, options = {}) {
        if (!state.masterEnabled || !state.sfxEnabled) return;
        if (!initAudioContext()) return;

        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const now = audioCtx.currentTime;

        // Rate limiting: avoid spamming identical SFX within 35ms
        if (now - lastSfxTime < 0.035 && type === lastSfxType) return;
        if (activeVoiceCount >= MAX_POLYPHONY) return;

        lastSfxTime = now;
        lastSfxType = type;
        activeVoiceCount++;

        try {
            switch (type) {
                // Navigation click: clean high-tech micro-tone
                case 'nav': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(1100, now);
                    osc.frequency.exponentialRampToValueAtTime(600, now + 0.04);

                    gain.gain.setValueAtTime(0.05, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.04);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.045);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 50);
                    break;
                }

                // Standard Button click: short tactile click
                case 'click': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.setValueAtTime(780, now);
                    osc.frequency.exponentialRampToValueAtTime(280, now + 0.03);

                    gain.gain.setValueAtTime(0.07, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.03);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.035);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 40);
                    break;
                }

                // Primary CTA button confirmation: dual-tone with low-mid body
                case 'cta': {
                    const oscLow = audioCtx.createOscillator();
                    const oscHigh = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();

                    oscLow.type = 'sine';
                    oscLow.frequency.setValueAtTime(220, now);
                    oscLow.frequency.exponentialRampToValueAtTime(140, now + 0.14);

                    oscHigh.type = 'sine';
                    oscHigh.frequency.setValueAtTime(660, now);
                    oscHigh.frequency.exponentialRampToValueAtTime(880, now + 0.08);

                    gain.gain.setValueAtTime(0.12, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.15);

                    oscLow.connect(gain);
                    oscHigh.connect(gain);
                    gain.connect(sfxBus);

                    oscLow.start(now);
                    oscHigh.start(now);
                    oscLow.stop(now + 0.16);
                    oscHigh.stop(now + 0.16);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 170);
                    break;
                }

                // Subtle button hover: ultra-subtle micro-tick
                case 'hover': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(2400, now);

                    gain.gain.setValueAtTime(0.015, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.012);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.014);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 20);
                    break;
                }

                // Spatial Chapter Transition Chime (Ch 1: 440, Ch 2: 523, Ch 3: 659, Ch 4: 784, Ch 5: 880)
                case 'chapter': {
                    const index = options.chapterIndex || 0;
                    const chordPitches = [440.0, 523.25, 659.25, 783.99, 880.0];
                    const baseFreq = chordPitches[Math.min(index, chordPitches.length - 1)];

                    const osc = audioCtx.createOscillator();
                    const harm = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();

                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(baseFreq, now);

                    harm.type = 'sine';
                    harm.frequency.setValueAtTime(baseFreq * 1.5, now); // Fifth harmonic

                    gain.gain.setValueAtTime(0.0001, now);
                    gain.gain.linearRampToValueAtTime(0.08, now + 0.05);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 1.4);

                    osc.connect(gain);
                    harm.connect(gain);
                    gain.connect(sfxBus);

                    osc.start(now);
                    harm.start(now);
                    osc.stop(now + 1.45);
                    harm.stop(now + 1.45);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 1500);
                    break;
                }

                // Spectroscopy Mode Selection: Tonal feedback
                case 'spectroscopy': {
                    const mode = options.mode || 'confocal';
                    let f1 = 880, f2 = 1320; // Confocal (Emerald 5th)
                    if (mode === 'darkfield') { f1 = 330; f2 = 440; } // Dark-Field (Sapphire 4th)
                    else if (mode === 'electron') { f1 = 1760; f2 = 2200; } // Electron Density (Cryo-EM ping)

                    const osc1 = audioCtx.createOscillator();
                    const osc2 = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();

                    osc1.type = 'sine';
                    osc1.frequency.setValueAtTime(f1, now);
                    osc2.type = 'sine';
                    osc2.frequency.setValueAtTime(f2, now);

                    gain.gain.setValueAtTime(0.07, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);

                    osc1.connect(gain);
                    osc2.connect(gain);
                    gain.connect(sfxBus);

                    osc1.start(now);
                    osc2.start(now);
                    osc1.stop(now + 0.23);
                    osc2.stop(now + 0.23);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 240);
                    break;
                }

                // Specimen structural layer toggle
                case 'layer': {
                    const checked = options.checked !== false;
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';

                    if (checked) {
                        osc.frequency.setValueAtTime(440, now);
                        osc.frequency.exponentialRampToValueAtTime(880, now + 0.06);
                    } else {
                        osc.frequency.setValueAtTime(660, now);
                        osc.frequency.exponentialRampToValueAtTime(330, now + 0.05);
                    }

                    gain.gain.setValueAtTime(0.06, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.065);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.07);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 80);
                    break;
                }

                // Modal dialog open (ambient whoosh)
                case 'modal-open': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(320, now);
                    osc.frequency.exponentialRampToValueAtTime(640, now + 0.16);

                    gain.gain.setValueAtTime(0.0001, now);
                    gain.gain.linearRampToValueAtTime(0.06, now + 0.04);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.19);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 200);
                    break;
                }

                // Modal dialog close (soft descending tuck)
                case 'modal-close': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(520, now);
                    osc.frequency.exponentialRampToValueAtTime(260, now + 0.12);

                    gain.gain.setValueAtTime(0.05, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.13);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 140);
                    break;
                }

                // Dropdown select change (liquid droplet click)
                case 'select': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(950, now);
                    osc.frequency.exponentialRampToValueAtTime(450, now + 0.04);

                    gain.gain.setValueAtTime(0.06, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.04);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.045);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 50);
                    break;
                }

                // Action Success (Harmonic major triad C5 -> E5 -> G5)
                case 'success': {
                    const notes = [523.25, 659.25, 783.99];
                    notes.forEach((freq, idx) => {
                        const noteTime = now + idx * 0.06;
                        const osc = audioCtx.createOscillator();
                        const gain = audioCtx.createGain();

                        osc.type = 'sine';
                        osc.frequency.setValueAtTime(freq, noteTime);

                        gain.gain.setValueAtTime(0.06, noteTime);
                        gain.gain.exponentialRampToValueAtTime(0.0001, noteTime + 0.28);

                        osc.connect(gain);
                        gain.connect(sfxBus);
                        osc.start(noteTime);
                        osc.stop(noteTime + 0.3);
                    });
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 400);
                    break;
                }

                // Action Error (Low dissonant pulse)
                case 'error': {
                    const osc1 = audioCtx.createOscillator();
                    const osc2 = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();

                    osc1.type = 'sawtooth';
                    osc1.frequency.setValueAtTime(180, now);
                    osc2.type = 'sawtooth';
                    osc2.frequency.setValueAtTime(195, now); // Dissonant beating

                    gain.gain.setValueAtTime(0.06, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.2);

                    osc1.connect(gain);
                    osc2.connect(gain);
                    gain.connect(sfxBus);

                    osc1.start(now);
                    osc2.start(now);
                    osc1.stop(now + 0.21);
                    osc2.stop(now + 0.21);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 220);
                    break;
                }

                // Dashboard view transition
                case 'tab': {
                    const osc = audioCtx.createOscillator();
                    const gain = audioCtx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(450, now);
                    osc.frequency.exponentialRampToValueAtTime(720, now + 0.06);

                    gain.gain.setValueAtTime(0.05, now);
                    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.07);

                    osc.connect(gain);
                    gain.connect(sfxBus);
                    osc.start(now);
                    osc.stop(now + 0.075);
                    setTimeout(() => activeVoiceCount = Math.max(0, activeVoiceCount - 1), 85);
                    break;
                }

                default:
                    activeVoiceCount = Math.max(0, activeVoiceCount - 1);
            }
        } catch (e) {
            activeVoiceCount = Math.max(0, activeVoiceCount - 1);
        }
    }

    // ─── 4. Audio Controls & Settings Management ───────────────────────────────
    function setMaster(enabled) {
        state.masterEnabled = !!enabled;
        localStorage.setItem(STORAGE_KEY_MASTER, state.masterEnabled);

        if (masterGain && audioCtx) {
            const now = audioCtx.currentTime;
            masterGain.gain.cancelScheduledValues(now);
            if (state.masterEnabled) {
                masterGain.gain.linearRampToValueAtTime(state.volume, now + 0.2);
                if (state.musicEnabled && !state.isSoundtrackPlaying) {
                    startSoundtrack();
                }
            } else {
                masterGain.gain.linearRampToValueAtTime(0.0001, now + 0.2);
            }
        } else if (state.masterEnabled) {
            startSoundtrack();
        }

        updateUI();
    }

    function setMusic(enabled) {
        state.musicEnabled = !!enabled;
        localStorage.setItem(STORAGE_KEY_MUSIC, state.musicEnabled);

        if (musicBus && audioCtx) {
            const now = audioCtx.currentTime;
            musicBus.gain.cancelScheduledValues(now);
            if (state.musicEnabled && state.masterEnabled) {
                if (!state.isSoundtrackPlaying) startSoundtrack();
                else musicBus.gain.linearRampToValueAtTime(0.35, now + 0.4);
            } else {
                musicBus.gain.linearRampToValueAtTime(0.0001, now + 0.4);
            }
        } else if (state.musicEnabled && state.masterEnabled) {
            startSoundtrack();
        }

        updateUI();
    }

    function setSfx(enabled) {
        state.sfxEnabled = !!enabled;
        localStorage.setItem(STORAGE_KEY_SFX, state.sfxEnabled);

        if (sfxBus && audioCtx) {
            const now = audioCtx.currentTime;
            sfxBus.gain.cancelScheduledValues(now);
            sfxBus.gain.linearRampToValueAtTime(state.sfxEnabled ? 0.65 : 0.0001, now + 0.1);
        }

        updateUI();
    }

    function setVolume(val) {
        const vol = Math.max(0, Math.min(1, parseFloat(val)));
        state.volume = vol;
        localStorage.setItem(STORAGE_KEY_VOLUME, vol);

        if (masterGain && audioCtx && state.masterEnabled) {
            const now = audioCtx.currentTime;
            masterGain.gain.cancelScheduledValues(now);
            masterGain.gain.linearRampToValueAtTime(vol, now + 0.08);
        }

        updateUI();
    }

    function toggleSettingsPanel() {
        const panel = document.getElementById('amrAudioPanel');
        if (!panel) return;
        const isOpen = panel.classList.contains('open');
        if (isOpen) {
            closeSettingsPanel();
        } else {
            openSettingsPanel();
        }
    }

    function openSettingsPanel() {
        const panel = document.getElementById('amrAudioPanel');
        if (!panel) return;
        panel.classList.add('open');
        playSfx('modal-open');
    }

    function closeSettingsPanel() {
        const panel = document.getElementById('amrAudioPanel');
        if (!panel) return;
        panel.classList.remove('open');
        playSfx('modal-close');
    }

    function acceptWelcomeSound(enabled) {
        state.promptDismissed = true;
        localStorage.setItem(STORAGE_KEY_DISMISSED, 'true');

        const card = document.getElementById('amrAudioWelcomeCard');
        if (card) {
            card.classList.add('fade-out');
            setTimeout(() => card.remove(), 600);
        }

        if (enabled) {
            setMaster(true);
            setMusic(true);
            setSfx(true);
            startSoundtrack();
            playSfx('cta');
        } else {
            setMaster(false);
            setMusic(false);
        }
    }

    // ─── 5. UI Synchronizer & Visual State ──────────────────────────────────────
    function updateUI() {
        const btn = document.getElementById('btnAudioToggle');
        const icon = document.getElementById('audioWavesIcon');
        const label = document.getElementById('audioToggleLabel');
        const volDisplay = document.getElementById('audioVolumeDisplay');
        const volRange = document.getElementById('audioVolumeRange');
        const pillVol = document.getElementById('audioPillVol');

        const masterSw = document.getElementById('audioMasterSwitch');
        const musicSw = document.getElementById('audioMusicSwitch');
        const sfxSw = document.getElementById('audioSfxSwitch');

        if (masterSw) masterSw.checked = state.masterEnabled;
        if (musicSw) musicSw.checked = state.musicEnabled;
        if (sfxSw) sfxSw.checked = state.sfxEnabled;

        const volPercent = Math.round(state.volume * 100);
        if (volDisplay) volDisplay.textContent = `${volPercent}%`;
        if (volRange) volRange.value = volPercent;
        if (pillVol) pillVol.textContent = state.masterEnabled ? `${volPercent}%` : 'Muted';

        if (label) {
            label.textContent = state.masterEnabled ? (state.musicEnabled ? 'Audio' : 'SFX Only') : 'Muted';
        }

        const isPlaying = state.masterEnabled && state.musicEnabled && state.isSoundtrackPlaying && state.isTabVisible;
        if (btn) {
            btn.classList.toggle('active', state.masterEnabled);
            btn.classList.toggle('music-playing', isPlaying);
        }
        if (icon) {
            icon.classList.toggle('active', isPlaying);
        }
    }

    // ─── 6. Global Interaction Event Listeners ──────────────────────────────────
    function setupInteractionListeners() {
        // Global Click Listener with Intelligent Delegation
        document.addEventListener('click', (e) => {
            const target = e.target.closest('button, a, select, input[type="checkbox"], .chapter-nav-item, .nav-item, .geo-feature-item, .nexus-shortcut');
            if (!target) return;

            // Prevent intercepting the audio settings panel controls themselves
            if (target.closest('#amrAudioPanel') || target.closest('#amrAudioWelcomeCard')) return;

            // CTA Primary Buttons
            if (target.classList.contains('hero-btn-primary') ||
                target.classList.contains('nexus-launch-btn') ||
                target.id === 'btnRunPrediction' ||
                target.id === 'btnExportPdf' ||
                target.classList.contains('btn-primary')) {
                playSfx('cta');
                return;
            }

            // Navigation Links
            if (target.classList.contains('geo-nav-link') ||
                target.classList.contains('chapter-nav-item') ||
                target.classList.contains('nav-item')) {
                playSfx('nav');
                return;
            }

            // Spectroscopy Mode Toggles
            if (target.classList.contains('hud-toggle-btn')) {
                let mode = 'confocal';
                if (target.textContent.includes('Dark-Field')) mode = 'darkfield';
                else if (target.textContent.includes('Electron')) mode = 'electron';
                playSfx('spectroscopy', { mode });
                return;
            }

            // Checkbox Layer Toggles
            if (target.tagName === 'INPUT' && target.type === 'checkbox') {
                playSfx('layer', { checked: target.checked });
                return;
            }

            // Standard Button Click
            playSfx('click');
        }, { passive: true });

        // Global Hover Listener for Primary CTAs and Navigation
        let hoverThrottle = false;
        document.addEventListener('mouseover', (e) => {
            if (hoverThrottle) return;
            const target = e.target.closest('.hero-btn-primary, .hero-btn-secondary, .geo-nav-link, .geo-auth-pill, .nexus-launch-btn');
            if (!target) return;

            hoverThrottle = true;
            playSfx('hover');
            setTimeout(() => hoverThrottle = false, 75);
        }, { passive: true });

        // Dropdown Selection Change
        document.addEventListener('change', (e) => {
            if (e.target.tagName === 'SELECT') {
                playSfx('select');
            }
        }, { passive: true });

        // Close Audio Settings Panel when clicking outside
        document.addEventListener('click', (e) => {
            const wrapper = document.getElementById('amrAudioControlWrapper');
            const panel = document.getElementById('amrAudioPanel');
            if (wrapper && panel && panel.classList.contains('open')) {
                if (!wrapper.contains(e.target)) {
                    closeSettingsPanel();
                }
            }
        });

        // Tab Visibility Change (smoothly mute when hidden, restore when visible)
        document.addEventListener('visibilitychange', () => {
            state.isTabVisible = !document.hidden;
            if (masterGain && audioCtx) {
                const now = audioCtx.currentTime;
                masterGain.gain.cancelScheduledValues(now);
                if (state.isTabVisible && state.masterEnabled) {
                    masterGain.gain.linearRampToValueAtTime(state.volume, now + 0.8);
                } else {
                    masterGain.gain.linearRampToValueAtTime(0.0001, now + 0.5);
                }
            }
            updateUI();
        });
    }

    // ─── 7. Welcome Card Display Logic ──────────────────────────────────────────
    function checkFirstVisitWelcome() {
        if (!state.promptDismissed) {
            const card = document.getElementById('amrAudioWelcomeCard');
            if (card) {
                setTimeout(() => {
                    card.classList.add('visible');
                }, 1200);
            }
        } else if (state.masterEnabled && state.musicEnabled) {
            // If already enabled on previous visit, unlock soundtrack on very first user interaction
            const unlockOnInteraction = () => {
                startSoundtrack();
                window.removeEventListener('click', unlockOnInteraction);
                window.removeEventListener('keydown', unlockOnInteraction);
                window.removeEventListener('scroll', unlockOnInteraction);
            };
            window.addEventListener('click', unlockOnInteraction, { once: true });
            window.addEventListener('keydown', unlockOnInteraction, { once: true });
            window.addEventListener('scroll', unlockOnInteraction, { once: true });
        }
    }

    // ─── 8. Global API Export ───────────────────────────────────────────────────
    window.AMRAudio = {
        // Core Control
        init: initAudioContext,
        startSoundtrack: startSoundtrack,
        stopSoundtrack: stopSoundtrack,
        toggle: () => setMaster(!state.masterEnabled),
        setMaster: setMaster,
        setMusic: setMusic,
        setSfx: setSfx,
        setVolume: setVolume,
        toggleSettingsPanel: toggleSettingsPanel,
        openSettingsPanel: openSettingsPanel,
        closeSettingsPanel: closeSettingsPanel,
        acceptWelcomeSound: acceptWelcomeSound,

        // Direct SFX Triggers
        playSfx: playSfx,
        chime: (freq = 520, duration = 1.2) => playSfx('chapter', { chapterIndex: Math.floor((freq - 440) / 80) }),
        navClick: () => playSfx('nav'),
        btnClick: () => playSfx('click'),
        ctaClick: () => playSfx('cta'),
        btnHover: () => playSfx('hover'),
        chapterTransition: (idx) => playSfx('chapter', { chapterIndex: idx }),
        spectroscopyChange: (mode) => playSfx('spectroscopy', { mode }),
        layerToggle: (checked) => playSfx('layer', { checked }),
        modalOpen: () => playSfx('modal-open'),
        modalClose: () => playSfx('modal-close'),
        actionSuccess: () => playSfx('success'),
        actionError: () => playSfx('error'),
        tabSwitch: () => playSfx('tab'),

        // State Getters
        get isPlaying() { return state.masterEnabled && state.musicEnabled && state.isSoundtrackPlaying; },
        get state() { return { ...state }; }
    };

    // ─── DOM Ready Initialization ───────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            setupInteractionListeners();
            updateUI();
            checkFirstVisitWelcome();
        });
    } else {
        setupInteractionListeners();
        updateUI();
        checkFirstVisitWelcome();
    }

})();
