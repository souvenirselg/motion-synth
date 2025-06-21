import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
from utils import sine_wave, saw_wave, square_wave, quantize_freq, SCALES
from synth_state import SynthState
import threading

# Butterworth lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)

# Delay effect class
""" class DelayEffect:
    def __init__(self, sample_rate, delay_time=0.3, feedback=0.4):
        self.sample_rate = sample_rate
        self.delay_samples = int(delay_time * sample_rate)
        self.feedback = feedback
        self.buffer = np.zeros(self.delay_samples * 10)  # 10x delay buffer for safety
        self.pos = 0

    def process(self, input_signal):
        output = np.zeros_like(input_signal)
        for i in range(len(input_signal)):
            delayed_sample = self.buffer[self.pos]
            output[i] = input_signal[i] + delayed_sample * self.feedback
            self.buffer[self.pos] = output[i]
            self.pos = (self.pos + 1) % len(self.buffer)
        return output """

# Audio engine class with streaming and DSP
class AudioEngine:
    def __init__(self, synth_state: SynthState):
        self.synth_state = synth_state
        self.sample_rate = synth_state.sample_rate
        self.block_size = 512
        self.time_pos = 0
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            callback=self.audio_callback)
        self.lock = threading.Lock()

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            t = (np.arange(frames) + self.time_pos) / self.sample_rate
            self.time_pos += frames
            waveform = np.zeros(frames)
            freqs = self.synth_state.notes
            adsr = self.synth_state.adsr

            # If no notes, output silence
            if len(freqs) == 0 or self.synth_state.volume < 0.01:
                outdata[:] = np.zeros((frames, 1))
                return

            # Generate waveform sum
            for freq in freqs:
                # Quantize freq to scale
                freq_q = quantize_freq(freq, SCALES[self.synth_state.scale], self.synth_state.root_midi)

                if self.synth_state.waveform == "sine":
                    waveform += sine_wave(freq_q, t)
                elif self.synth_state.waveform == "saw":
                    waveform += saw_wave(freq_q, t)
                elif self.synth_state.waveform == "square":
                    waveform += square_wave(freq_q, t)

            waveform /= max(len(freqs), 1)

            # Apply ADSR envelope (simplified for continuous tone)
            dt = frames / self.sample_rate
            envelope = np.array([adsr.process(1 / self.sample_rate) for _ in range(frames)])
            waveform *= envelope

            # Apply volume
            waveform *= self.synth_state.volume

            # Apply delay effect
            #waveform = self.delay.process(waveform)

            # Apply lowpass filter for smoothness
            waveform = lowpass_filter(waveform, cutoff=1000, fs=self.sample_rate)

            # Prevent clipping
            waveform = np.clip(waveform, -1, 1)

            outdata[:] = waveform.reshape(-1, 1)
