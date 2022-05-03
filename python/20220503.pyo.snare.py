from pyo import *
import numpy as np

s = Server().boot()

class Track:
    def __init__(self, name, root, beat, note, speed, mul):
        self.name = name
        self.root = root
        self.beat = beat
        self.note = note
        self.speed = speed

        self.mul = Sig(mul)
        self.mul.ctrl(title=name + " Gain")

class Maestro:
    def __init__(self, tracks: [Track], time=0.25):
        self.tracks = tracks
        self.time = time
        self.idx = 0

        self.clock = Pattern(self.tick, time=self.time)

    def start(self):
        self.clock.play()

    def tick(self):
        for t in self.tracks:
            beat = t.beat[self.idx % len(t.beat)]
            note = t.note[self.idx % len(t.note)]
            if beat:
                t.play(t.root + note)
        self.idx = self.idx + 1

class Snare (Track):
    def __init__(self, name="Snare", beat=[1], speed=1, mul=1,
                 dur=0.25) -> None:
        super().__init__(name, 0, beat, [0], speed, mul)
        self.peakFreq = 400

        self.trig = Trig()

        self.ampenv = LinTable([(0,0.0000),(733,0.1091),(1484,0.1091),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(8192,0.0667)])
        self.noiseenv = LinTable([(0,0.0000),(357,0.9091),(1609,0.0061)])
        self.revenv = LinTable([(0,0.0000),(340,1.0000),(1627,0.1091),(8192,0.0000)])
        self.ampenv.graph(title=self.name + " Punch Amplitude")
        self.pitchenv.graph(title=self.name + " Punch Pitch")
        self.noiseenv.graph(title=self.name + " Noise Amplitude")
        self.revenv.graph(title=self.name + " Noise Reverb")

        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=dur, mul=self.peakFreq)
        self.noiseamp = TrigEnv(self.trig, table=self.noiseenv, dur=dur, mul=self.mul)
        self.revamp = TrigEnv(self.trig, table=self.revenv, dur=dur, mul=1)

        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

        self.noise = PinkNoise(mul=self.noiseamp).mix(2)
        self.filter = ButBP(self.noise, freq=1000, q=2)
        self.filter.ctrl()

        self.effect = Freeverb(self.filter, size=1, damp=.5, bal=self.revamp)
        self.effect.ctrl()

        self.effect.out()

    def play(self, note=0):
        self.trig.play()

M = Maestro(time=0.125, tracks=[
        Snare(name="Snare", mul = 0.4, dur = 0.2,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 0, 0, 1, 0, 0, 0 ]
        ),
])

s.start()
M.start()
s.gui(locals())
