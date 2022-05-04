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

        self.ampenv = LinTable([(0,0.0000),(554,0.0667),(1377,0.0485),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(787,0.1807),(8192,0.0000)])
        self.noisepitchenv = LinTable([(0,0.0000),(340,1.0000),(787,0.1807),(8192,0.0000)])
        self.noiseenv = LinTable([(0,0.0000),(178,1.0000),(2325,0.0000)])
        self.revenv = LinTable([(0,0.0000),(840,0.9939),(2271,0.9152),(3219,0.2061),(5616,0.0000)])
        self.ampenv.graph(title=self.name + " Punch Amplitude")
        self.pitchenv.graph(title=self.name + " Punch Pitch")
        self.noisepitchenv.graph(title=self.name + " Noise Pitch")
        self.noiseenv.graph(title=self.name + " Noise Amplitude")
        self.revenv.graph(title=self.name + " Noise Reverb")

        gain = 10

        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=dur, mul=gain*self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=dur, mul=self.peakFreq)
        self.noisepitch = TrigEnv(self.trig, table=self.noisepitchenv, dur=dur, mul=3000)
        self.noiseamp = TrigEnv(self.trig, table=self.noiseenv, dur=dur, mul=gain*self.mul)
        self.revamp = TrigEnv(self.trig, table=self.revenv, dur=dur, mul=1)

        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

        self.noise = PinkNoise(mul=self.noiseamp).mix(2)
        self.filter = ButBP(self.noise, freq=self.noisepitch, q=2)
        self.filter.ctrl()

        self.effect = Freeverb(self.filter, size=.2, damp=.5, bal=self.revamp)
        self.effect.ctrl()

        self.effect.out()

    def play(self, note=0):
        self.trig.play()

class KickBeat (Track):
    def __init__(self, name="Kick", root=49, beat=[1], note=[0], speed=1, mul=1,
                 dur=0.25) -> None:
        super().__init__(name, root, beat, note, speed, mul)
        self.peakFreq = 400

        self.trig = Trig()
        self.ampenv = LinTable([(0,0.0000),(600,1.0000),(6564,0.6788),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(8192,0.0667)])
        # self.ampenv.graph(title="Kick Amplitude")
        # self.pitchenv.graph(title="Kick Pitch")
        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=dur, mul=self.peakFreq)
        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

    def play(self, note=0):
        self.trig.play()

class Piou (Track):
    def __init__(self, name="Piou", root=49, beat=[1], note=[0], speed=1, mul=1,
                 dur=0.25) -> None:
        super().__init__(name, root, beat, note, speed, mul)
        self.maxFreq = 8000
        self.minFreq = 10

        self.trig = Trig()
        self.ampenv = LinTable([(0,0.0000),(600,1.0000),(6564,0.6788),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(930,0.2667),(5097,0.0000)])
        self.ampenv.graph(title="Piou Amplitude")
        self.pitchenv.graph(title="Piou Pitch")
        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=dur, mul=self.maxFreq, add=self.minFreq)
        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

    def play(self, note=0):
        self.trig.play()

class BassBeat (Track):
    def __init__(self, name="Bass Beat", root=49, beat=[1], note=[0], speed=1, mul=1,
                 attack=0.016, sustain=0.5, dur=0.2):
        super().__init__(name, root, beat, note, speed, mul)

        self.env = Adsr(attack=attack, decay=attack, sustain=sustain, release=2*attack, dur=dur, mul=self.mul)
        # self.env.ctrl()

        self.table = HarmTable([.8, 1, .8 , .5, .3, .2, .1, .05])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])
        self.osc.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.env.play()

class Peak:
  def __init__(self, note=48, mul=1.0) -> None:
    self.noise1 = PinkNoise()
    self.noise2 = PinkNoise()
    self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=14, type=2, stages=2, mul=mul).out()

time = 0.09
root = 25

# P = Peak(note=root+60, mul=0.15)
# Q = Peak(note=root+67, mul=0.05)

M = Maestro(time=time, tracks=[
        BassBeat(name="Bass Beat", mul = 0.1, root = root, dur=time, attack=0.2*time, sustain=1,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 1, 1, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        BassBeat(name="Bass Harm", mul = 0.012, root = root+24, dur=time, attack=0.2*time, sustain=1,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 1, 1, 1, 0 ],
            note = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ),
        KickBeat(name="Kick", mul = 0.4, dur = time,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 1, 1, 0, 0  ]
        ),
        Piou(name="Piou", mul = 0.085, dur = time,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0 ]
        ),
        # Snare(name="Snare", mul = 0.1, dur = time,
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 0, 0, 1, 0 ]
        # ),
])

s.start()
M.start()
s.gui(locals())
