from pyo import *
import numpy as np

s = Server().boot()

class Track:
    def __init__(self, root, beat, note, speed, mul):
        self.root = root
        self.beat = beat
        self.note = note
        self.speed = speed

        self.mul = Sig(mul)
        self.mul.ctrl(title="Amplitude")

class Maestro:
    def __init__(self, tracks: [Track], time=0.25):
        self.tracks = tracks
        self.time = time
        self.idx = 0

        self.clock = Pattern(self.tick, time=self.time).play()

    def tick(self):
        for t in self.tracks:
            beat = t.beat[self.idx % len(t.beat)]
            note = t.note[self.idx % len(t.note)]
            if beat:
                t.play(t.root + note)
        self.idx = self.idx + 1

class BassBeat (Track):
    def __init__(self, root=49, beat=[1], note=[0], speed=1, mul=1,
                 attack=0.016, sustain=0.5, dur=0.2):
        super().__init__(root, beat, note, speed, mul)

        self.env = Adsr(attack=attack, decay=attack, sustain=sustain, release=2*attack, dur=dur, mul=self.mul)
        self.env.ctrl()

        self.table = HarmTable([.8, 1, .8 , .5, .3, .2, .1, .05])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])
        self.osc.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.env.play()

class GuitarPlucking (Track):
    def __init__(self, root=49, beat=[1], note=[0], speed=1, mul=1,
                 dur=0.2):
        super().__init__(root, beat, note, speed, mul)

        self.envAmp = Linseg([(0.0000,0.0000),(0.0121,0.1646),(0.0247,0.0709),(0.1284,0.0000)], mul=self.mul)
        self.envAmp.graph(title="Guitar Plucking Amplitude")

        self.envRev = Linseg([(0.0000,0.0000),(0.0121,0.7370),(0.1074,0.5285),(0.1284,0.0000)], mul=self.mul)
        self.envRev.graph(title="Guitar Plucking Reverb")

        self.table = HarmTable([1])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.envAmp, phase=[0,.4])

        self.effect = Freeverb(self.osc, size=.8, damp=.5, bal=self.envRev)
        self.effect.ctrl()
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.envAmp.play()
        self.envRev.play()

class BassPlucking (Track):
    def __init__(self, root=49, beat=[1], note=[0], speed=1, mul=1,
                 reverb=0, dur=0.2):
        super().__init__(root, beat, note, speed, mul)

        self.envAmp = Linseg([(0.0000,0.0000),(0.0118,0.0648),(0.0634,0.0509),(0.1284,0.0000)], mul=self.mul)
        self.envAmp.graph(title="Bass Plucking Amplitude")

        self.envRev = Linseg([(0.0000,0.0000),(0.0121,0.7370),(0.0247,0.1429),(0.1284,0.0000)], mul=self.mul)
        self.envRev.graph(title="Bass Plucking Reverb")

        self.table = HarmTable([.8, 1, .8])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.envAmp, phase=[0,.4])

        self.effect = Freeverb(self.osc, size=.8, damp=.5, bal=self.envRev)
        self.effect.ctrl()
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.envAmp.play()
        self.envRev.play()

class KickBeat (Track):
    def __init__(self, root=49, beat=[1], note=[0], speed=1, mul=1,
                 dur=0.25) -> None:
        super().__init__(root, beat, note, speed, mul)
        self.peakFreq = 400

        self.trig = Trig()
        self.ampenv = LinTable([(0,0.0000),(600,1.0000),(6564,0.6788),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(8192,0.0667)])
        self.ampenv.graph(title="Kick Amplitude")
        self.pitchenv.graph(title="Kick Pitch")
        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=dur, mul=self.peakFreq)
        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

    def play(self, note=0):
        self.trig.play()

M = Maestro(time=0.125, tracks=[
        BassBeat(mul = 0.2, root = 25, attack=0.02, sustain=0.5,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        GuitarPlucking(mul = 0.15, root = 73,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0 ]
        ),
        BassPlucking(mul = 0.6, root = 37,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0 ]
        ),
        BassPlucking(mul = 0.2, root = 37,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0 ]
        ),
        KickBeat(mul = 0.3, root = 49, dur = 0.2,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 ]
        ),
])

s.start()
s.gui(locals())
