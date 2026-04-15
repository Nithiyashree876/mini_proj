"""
Audio Capture Module.
Records audio from the microphone using sounddevice, and provides
synthetic audio generation for demo/testing purposes.
"""

import numpy as np
import io
import wave
import os


class AudioCapture:
    """Audio capture from microphone with synthetic generation for demos."""

    def __init__(self, sample_rate=None, duration=None):
        if sample_rate is None or duration is None:
            from utils.config import AUDIO_SAMPLE_RATE, AUDIO_DURATION
            sample_rate = sample_rate or AUDIO_SAMPLE_RATE
            duration = duration or AUDIO_DURATION
        self.sample_rate = sample_rate
        self.duration = duration
        self._sounddevice = None

    def _get_sounddevice(self):
        """Lazy-load sounddevice to avoid import errors when not needed."""
        if self._sounddevice is None:
            try:
                import sounddevice as sd
                self._sounddevice = sd
            except ImportError:
                raise ImportError(
                    "sounddevice is required for audio capture. "
                    "Install it with: pip install sounddevice"
                )
        return self._sounddevice

    def record(self, duration=None):
        """Record audio from the default microphone.

        Args:
            duration: Recording duration in seconds (uses default if None)

        Returns:
            1D numpy array of float32 audio samples
        """
        sd = self._get_sounddevice()
        duration = duration or self.duration

        print(f"  Recording for {duration} seconds...")
        
        try:
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Block until recording is complete
        except Exception as e:
            # Catch device -1 error (no default mic found) or driver errors
            raise RuntimeError("No microphone detected! Please plug in a microphone or check your computer's privacy settings.")

        print("  Recording complete.")

        return audio_data.flatten()

    def record_to_buffer(self, duration=None):
        """Record audio and return as a WAV bytes buffer.

        Args:
            duration: Recording duration in seconds

        Returns:
            io.BytesIO containing WAV data
        """
        audio = self.record(duration)

        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        buffer.seek(0)
        return buffer

    def get_available_devices(self):
        """List available audio input/output devices."""
        sd = self._get_sounddevice()
        return sd.query_devices()

    @staticmethod
    def generate_synthetic_audio(duration=3, sample_rate=16000, base_freq=150, seed=None):
        """Generate synthetic audio that simulates speech-like patterns.

        Creates realistic audio with fundamental frequency, harmonics,
        formant resonances, amplitude envelope, and noise — suitable
        for testing the speaker recognition pipeline.

        Args:
            duration: Length in seconds
            sample_rate: Sample rate in Hz
            base_freq: Fundamental frequency (varies per simulated person)
            seed: Random seed for reproducibility

        Returns:
            1D float32 numpy array
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.linspace(0, duration, int(duration * sample_rate), dtype=np.float32)

        # Fundamental frequency with vibrato
        vibrato = 5 * np.sin(2 * np.pi * 5 * t)
        signal = np.sin(2 * np.pi * (base_freq + vibrato) * t)

        # Add harmonics (characteristic of human voice)
        for harmonic in [2, 3, 4, 5]:
            amplitude = 1.0 / (harmonic ** 1.5)
            freq_variation = np.random.uniform(0.98, 1.02)
            signal += amplitude * np.sin(
                2 * np.pi * base_freq * harmonic * freq_variation * t
            )

        # Add formant resonances (vocal tract characteristics)
        formants = [800, 1200, 2500]
        for f in formants:
            f_var = f * np.random.uniform(0.9, 1.1)
            signal += 0.3 * np.sin(2 * np.pi * f_var * t) * np.exp(-t * 0.5)

        # Speech-like amplitude envelope (syllable rhythm)
        envelope = np.ones_like(t)
        num_syllables = int(duration * 3)
        for _ in range(num_syllables):
            center = np.random.uniform(0, duration)
            width = np.random.uniform(0.1, 0.3)
            envelope += 0.5 * np.exp(-((t - center) ** 2) / (2 * width ** 2))

        signal *= envelope

        # Add ambient noise
        signal += 0.02 * np.random.randn(len(signal)).astype(np.float32)

        # Normalize to [-0.9, 0.9]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.9

        return signal.astype(np.float32)
