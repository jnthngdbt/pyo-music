from pyo import *
import numpy as np

s = Server().boot()

class Track:
    def __init__(self, name="Track", root=49, note=[0], beat=[1], dur=.2, prob=.6, speed=1, mul=1):
        self.name = name
        self.root = root
        self.beat = beat
        self.note = note
        self.dur = dur
        self.prob = prob
        self.speed = speed

        self.mul = Sig(mul)
        self.mul.ctrl(title=name + " Gain")

class Maestro:
    def __init__(self, tracks: [Track], time=0.25, nbSectionsToggle=0):
        self.tracks = tracks
        self.time = time
        self.nbSectionsToggle = nbSectionsToggle

        self.idx = 0
        self.isShifted = False

        self.muls = [t.mul.value for t in self.tracks]
        self.sectionSize = max([max([len(t.note), len([t.beat])]) for t in self.tracks])

        self.clock = Pattern(self.tick, time=self.time).play()

    def tick(self):
        doToggle = (self.nbSectionsToggle > 0) and (self.idx % (self.sectionSize * self.nbSectionsToggle) == 0)
        if doToggle:
            self.toggleTracks()

        if doToggle and (self.nbTracksActive >= len(self.tracks)):
            self.shiftRoot()

        for t in self.tracks:
            beat = t.beat[self.idx % len(t.beat)]
            note = t.note[self.idx % len(t.note)]
            if beat:
                t.play(t.root + note)

        self.idx = self.idx + 1

    def toggleTracks(self):
        self.nbTracksActive = 0
        for i, t in enumerate(self.tracks):
            if random.random() < t.prob:
                t.mul.value = self.muls[i]
                self.nbTracksActive = self.nbTracksActive + 1
            else:
                t.mul.value = 0

        if self.nbTracksActive == 0:
            i = random.choice(range(len(self.tracks)))
            self.tracks[i].mul.setValue(self.muls[i])

    def shiftRoot(self):
        s = -1 if self.isShifted else 1
        for t in self.tracks:
            t.root = t.root + s
        self.isShifted = ~self.isShifted

class Snare (Track):
    def __init__(self, cutoff=2000, **kwargs):
        super().__init__(**kwargs)
        self.peakFreq = 400

        self.trig = Trig()

        self.ampenv = LinTable([(0,0.0000),(232,0.5),(733,0.1455),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(1001,0.1818),(8192,0.0667)])
        self.noiseenv = ExpTable([(0,0.0000),(125,1.0000),(7887,0.0242),(8192,0.0000)])
        self.ampenv.graph(title=self.name + " Punch Amplitude")
        # self.pitchenv.graph(title=self.name + " Punch Pitch")
        self.noiseenv.graph(title=self.name + " Noise Amplitude")

        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=self.dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=self.dur, mul=self.peakFreq)
        self.noiseamp = TrigEnv(self.trig, table=self.noiseenv, dur=self.dur, mul=self.mul)

        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

        self.noise = Noise(mul=self.noiseamp).mix(2)
        self.lp = ButLP(self.noise, freq=cutoff)
        self.lp.ctrl()
        self.lp.out()

    def play(self, note=0):
        self.trig.play()

class SnarePunchy (Snare):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noiseenv.list = [(0,0.0000),(125,1.0000),(2575,0.1030),(8192,0.0000)]

class Kick (Track):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.peakFreq = 400

        self.trig = Trig()
        self.ampenv = LinTable([(0,0.0000),(600,1.0000),(6564,0.6788),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(8192,0.0667)])
        # self.ampenv.graph(title="Kick Amplitude")
        # self.pitchenv.graph(title="Kick Pitch")
        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=self.dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=self.dur, mul=self.peakFreq)
        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

    def play(self, note=0):
        self.trig.play()

class BassBeat (Track):
    def __init__(self, attack=0.016, sustain=0.5, tone=[.8, 1., .8, .5, .3, .2, .1], reverb=0., **kwargs):
        super().__init__(**kwargs)

        self.env = Adsr(attack=attack, decay=attack, sustain=sustain, release=2*attack, dur=self.dur, mul=self.mul)
        # self.env.ctrl()

        self.table = HarmTable(tone)
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])
        self.effect = Freeverb(self.osc, size=0.8, damp=0.5, bal=reverb)
        self.effect.out()

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

M = Maestro(time=time, nbSectionsToggle=2, tracks=[
        BassBeat(name="Bass Beat", mul = 0.1, root = root, dur=time, attack=0.2*time, sustain=1, tone=[.8, 1, 1, .8, .3, .2, .1],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 1, 1, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        BassBeat(name="Bass Harm", mul = 0.012, root = root+24, dur=time, attack=0.2*time, sustain=1, tone=[.8, 1.,.8],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 1, 0, 1, 1, 1, 0 ],
            note = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ),
        BassBeat(name="Mid", mul = 0.004, root = root+48, dur=2*time, attack=0.2*time, sustain=1, tone = [1, .8], reverb=.4,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0 ],
            note = [ 7, 7, 7, 0, 0, 0, 7, 7, 7, 0, 0, 0, 7, 7, 0, 0]
        ),
        BassBeat(name="Mid High", mul = 0.004, root = root+48, dur=2*time, attack=0.2*time, sustain=1, tone = [1, .8], reverb=.4,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0 ],
            note = [12,12,12, 7, 7, 7,12,12,12, 7, 7, 0,12,12, 7, 7]
        ),
        Kick(name="Kick", mul = 0.4, dur = time,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 1, 1, 0, 0  ]
        ),
        SnarePunchy(name="Snare", mul = 0.5, dur = time, cutoff=800,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0 ]
        ),
])

s.start()
s.gui(locals())
