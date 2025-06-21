import numpy as np

# ADSR envelope class
class ADSR:
    def __init__(self, attack=0.05, decay=2, sustain=3, release=0.2, sample_rate=44100):
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.sample_rate = sample_rate
        self.state = "idle"
        self.time = 0.0

    def note_on(self):
        self.state = "attack"
        self.time = 0.0

    def note_off(self):
        self.state = "release"
        self.time = 0.0

    def process(self, dt):
        self.time += dt
        if self.state == "attack":
            if self.time >= self.attack:
                self.state = "decay"
                self.time = 0.0
                return 1.0
            return self.time / self.attack
        elif self.state == "decay":
            if self.time >= self.decay:
                self.state = "sustain"
                return self.sustain
            return 1 - (1 - self.sustain) * (self.time / self.decay)
        elif self.state == "sustain":
            return self.sustain
        elif self.state == "release":
            if self.time >= self.release:
                self.state = "idle"
                return 0.0
            return self.sustain * (1 - self.time / self.release)
        else:
            return 0.0

# Global synth state
class SynthState:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.waveform = "sine"  # sine, saw, square
        self.scale = "major"
        self.root_midi = 60  # Middle C
        self.adsr = ADSR(sample_rate=sample_rate)
        self.volume = 0.5
        self.notes = []  # active frequencies
