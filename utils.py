import math
import numpy as np

# Frequency / MIDI conversions
def freq_to_midi(freq):
    return 69 + 12 * math.log2(freq / 440)

def midi_to_freq(midi):
    return 440 * 2 ** ((midi - 69) / 12)

# Quantize frequency to nearest note in scale
def quantize_freq(freq, scale, root_midi=60):
    midi = freq_to_midi(freq)
    octave = int(midi // 12)
    note_in_octave = int(round(midi)) % 12
    closest_note = min(scale, key=lambda x: abs(x - note_in_octave))
    quantized_midi = octave * 12 + closest_note
    return midi_to_freq(quantized_midi)

# Common scales (semitones relative to root note)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
}

# Generate waveforms
def sine_wave(freq, t):
    return np.sin(2 * np.pi * freq * t)

def saw_wave(freq, t):
    return 2 * (t * freq - np.floor(0.5 + t * freq))

def square_wave(freq, t):
    return np.sign(np.sin(2 * np.pi * freq * t))
