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

        self.clock = Pattern(self.tick, time=self.time).play()

    def tick(self):
        for t in self.tracks:
            beat = t.beat[self.idx % len(t.beat)]
            note = t.note[self.idx % len(t.note)]
            if beat:
                t.play(t.root + note)
        self.idx = self.idx + 1

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

class GuitarPlucking (Track):
    def __init__(self, name="Guitar Pluck", root=49, beat=[1], note=[0], speed=1, mul=1,
                 dur=0.2):
        super().__init__(name, root, beat, note, speed, mul)

        self.envAmp = Linseg([(0.0000,0.0000),(0.0067,0.1646),(0.0631,0.1516),(0.1284,0.0000)], mul=self.mul)
        self.envAmp.graph(title=self.name + " Amplitude")

        self.envRev = Linseg([(0.0000,0.0000),(0.0121,0.7370),(0.1074,0.5285),(0.1284,0.0000)], mul=self.mul)
        # self.envRev.graph(title=self.name + " Reverb")

        self.table = HarmTable([1])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.envAmp, phase=[0,.4])

        self.effect = Freeverb(self.osc, size=.8, damp=.5, bal=self.envRev)
        # self.effect.ctrl()
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.envAmp.play()
        self.envRev.play()

class BassPlucking (Track):
    def __init__(self, name="Bass Pluck", root=49, beat=[1], note=[0], speed=1, mul=1,
                 reverb=0, dur=0.2):
        super().__init__(name, root, beat, note, speed, mul)

        self.envAmp = Linseg([(0.0000,0.0000),(0.0118,0.0648),(0.0634,0.0509),(0.1284,0.0000)], mul=self.mul)
        # self.envAmp.graph(title=self.name + " Amplitude")

        self.envRev = Linseg([(0.0000,0.0000),(0.0121,0.7370),(0.0247,0.1429),(0.1284,0.0000)], mul=self.mul)
        # self.envRev.graph(title=self.name + " Reverb")

        self.table = HarmTable([.8, 1])
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.envAmp, phase=[0,.4])

        self.effect = Freeverb(self.osc, size=.8, damp=.5, bal=self.envRev)
        # self.effect.ctrl()
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = midiToHz(note)
        self.envAmp.play()
        self.envRev.play()

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

root = 25
M = Maestro(time=0.125, tracks=[
        BassBeat(name="Bass Beat", mul = 0.2, root = root, attack=0.02, sustain=0.5,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        GuitarPlucking(name="Guitar High Pluck", mul = 0.05, root = root+48,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0 ]
        ),
        # BassPlucking(name="Bass Pluck Root", mul = 0.8, root = root+12,
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1 ],
        #     note = [ 0 ]
        # ),
        # BassPlucking(name="Bass Pluck 12 Beat", mul = 0.8, root = root+12,
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
        #     note = [ 12 ]
        # ),
        BassPlucking(name="Bass Pluck Harm Beat", mul = 0.8, root = root+12,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0 ]
        ),
        KickBeat(name="Kick", mul = 0.4, dur = 0.2,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 ]
        ),
        Snare(name="Snare", mul = 0.4, dur = 0.2,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 0, 0, 1, 0, 0, 0 ]
        ),
])

s.start()
s.gui(locals())
