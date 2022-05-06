from pyo import *
import numpy as np

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
    x = []
    notes = np.array(notes)
    for i in octaves:
        x = np.concatenate([x, i * 12 + notes], axis=0)
    return x.tolist()

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
    def __init__(self, tracks: [Track], time=0.25, nbSectionsToggle=2):
        self.tracks = tracks
        self.time = time
        self.nbSectionsToggle = nbSectionsToggle

        self.idx = 0
        self.isShifted = False

        self.muls = [t.mul.value for t in self.tracks]
        self.sectionSize = max([max([len(t.note), len([t.beat])]) for t in self.tracks])

        self.clock = Pattern(self.tick, time=self.time).play()

    def tick(self):
        doToggle = self.idx % (self.sectionSize * self.nbSectionsToggle) == 0
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
                t.mul.setValue(self.muls[i])
                self.nbTracksActive = self.nbTracksActive + 1
            else:
                t.mul.setValue(0)

        if self.nbTracksActive == 0:
            i = random.choice(range(len(self.tracks)))
            self.tracks[i].mul.setValue(self.muls[i])

    def shiftRoot(self):
        s = -1 if self.isShifted else 1
        for t in self.tracks:
            t.root = t.root + s
        self.isShifted = ~self.isShifted

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

class Snare (Track):
    def __init__(self, cutoff=2000, **kwargs):
        super().__init__(**kwargs)

        self.peakFreq = 400

        self.trig = Trig()

        self.ampenv = LinTable([(0,0.0000),(232,0.1455),(733,0.1455),(8191,0.0000)])
        self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(1001,0.1818),(8192,0.0667)])
        self.noiseenv = ExpTable([(0,0.0000),(125,1.0000),(7887,0.0242),(8192,0.0000)])
        # self.ampenv.graph(title=self.name + " Punch Amplitude")
        # self.pitchenv.graph(title=self.name + " Punch Pitch")
        # self.noiseenv.graph(title=self.name + " Noise Amplitude")

        self.amp = TrigEnv(self.trig, table=self.ampenv, dur=self.dur, mul=self.mul)
        self.pitch = TrigEnv(self.trig, table=self.pitchenv, dur=self.dur, mul=self.peakFreq)
        self.noiseamp = TrigEnv(self.trig, table=self.noiseenv, dur=self.dur, mul=self.mul)

        self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
        self.osc.out()

        self.noise = Noise(mul=self.noiseamp).mix(2)
        self.lp = ButLP(self.noise, freq=cutoff)
        self.lp.ctrl(title=self.name + "Cutoff")
        self.lp.out()

    def play(self, note=0):
        self.trig.play()

class SnarePunchy (Snare):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.noiseenv.list = [(0,0.0000),(125,1.0000),(2575,0.1030),(8192,0.0000)]
        self.ampenv.list = [(0,0.0000),(232,0.5),(733,0.1455),(8191,0.0000)]

class Arpeggio (Track):
    def __init__(self, notes=[0,3,5,7,10], sustain=0.5, doMirror=False, **kwargs):
        super().__init__(**kwargs)

        self.idx = 0
        self.notes = notes

        if doMirror:
          self.notes = self.notes + self.namenotes[-2:0:-1] # weird slicing: all except first and last, reversed

        self.table = HarmTable([1]) # , 0.3, 0.1, 0.02, 0.005
        self.osc = Osc(table=self.table, freq=[100,101])

        self.effect = Freeverb(self.osc, size=0.8, damp=0.5, bal=0.5)

        p = 0.012
        self.env = Adsr(attack=p, decay=p, sustain=sustain, release=p, dur=self.dur)

        self.out = self.mul * self.effect * self.env
        self.out.out()

    def play(self, note=0):
        note = self.root + self.notes[self.idx]
        f = midiToHz(note)
        self.osc.freq = [f, f+1]
        self.env.play()
        self.idx = (self.idx + 1) % len(self.notes)

class Peak:
  def __init__(self, note=48, mul=1.0) -> None:
    self.noise1 = PinkNoise()
    self.noise2 = PinkNoise()
    self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=14, type=2, stages=2, mul=mul).out()
