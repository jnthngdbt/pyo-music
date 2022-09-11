from attr import has
from pyo import *
import numpy as np

s = Server().boot()
s.start()

def randRange(a,b):
    r = b - a
    return a + r * np.random.rand()

class Base:
    def __init__(self, lfo=0.015, cutoff=20000, phase=Sig(.75), mul=1.):
        self.lfo = lfo
        self.cutoff = cutoff
        self.amp = Sig(mul)
        # self.amp.ctrl()
        self.phase = phase
        self.lfoMul = self.amp * Sine(freq=self.lfo * randRange(0.6, 1.5), phase=self.phase).range(0, 1)

class Pad (Base):
    def __init__(self, note, bw, damp, reverb=.3, **kwargs):
        super().__init__(**kwargs)

        padSize = 262144 * 2
        padFreq = 440
        padRatio = s.getSamplingRate() / padSize

        self.note = note

        self.table = PadSynthTable(
          basefreq=padFreq,
          size=padSize,
          spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
          bw=bw, # 20: org, 50: def strings, 70: dreamy more highs
          bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
          damp=damp, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
          nharms=64, # 64: def
        )

        f = midiToHz(note)
        freqRatio = padRatio * f / padFreq

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(self.table, freq=freqRatio))
        self.pipeline.append(Compress(self.pipeline[-1], thresh=-30., ratio=2.))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
        self.pipeline.append(Delay(self.pipeline[-1], delay=[0.10, 0.12], feedback=.5))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=.9, damp=.5, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class Sample (Base):
    def __init__(self, note, reverb=.3, **kwargs):
        super().__init__(**kwargs)

        self.note = note

        self.table = SndTable("C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\slider.spectrum.sample.wav")
        
        padFreq = 220 # A3: 220Hz, A4: 440Hz
        padSize = self.table.size
        padRatio = s.getSamplingRate() / padSize

        f = midiToHz(self.note)
        freqRatio = padRatio * f / padFreq

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(self.table, freq=freqRatio))
        self.pipeline.append(Compress(self.pipeline[-1], thresh=-30., ratio=2.))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
        # self.pipeline.append(Delay(self.pipeline[-1], delay=[0.10, 0.12], feedback=.5))
        # self.pipeline.append(Freeverb(self.pipeline[-1], size=.9, damp=.5, bal=reverb))
        self.pipeline.append(WGVerb(self.pipeline[-1], feedback=.5, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class Pad (Base):
    def __init__(self, note, bw, damp, reverb=.3, **kwargs):
        super().__init__(**kwargs)

        padSize = 262144 * 2
        padFreq = 440
        padRatio = s.getSamplingRate() / padSize

        self.note = note

        self.table = PadSynthTable(
          basefreq=padFreq,
          size=padSize,
          spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
          bw=bw, # 20: org, 50: def strings, 70: dreamy more highs
          bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
          damp=damp, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
          nharms=64, # 64: def
        )

        f = midiToHz(note)
        freqRatio = padRatio * f / padFreq

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(self.table, freq=freqRatio))
        self.pipeline.append(Compress(self.pipeline[-1], thresh=-30., ratio=2.))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
        self.pipeline.append(Delay(self.pipeline[-1], delay=[0.10, 0.12], feedback=.5))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=.9, damp=.5, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=self.cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class ToneBeatSimple (Base):
    def __init__(self, note, dur, decay=.1, sustain=.2, reverb=.3, tone=[1, 0.3, 0.1, 0.02, 0.005], **kwargs) -> None:
        super().__init__(**kwargs)

        f = midiToHz(note)

        self.table = HarmTable(tone)

        durValue = dur
        if hasattr(dur, 'value'):
            if hasattr(dur.value, 'value'):
                durValue = dur.value.value
            else:
                durValue = dur.value

        self.env = Adsr(attack=.01, decay=decay, sustain=sustain, release=.01, dur=durValue)

        self.lfoPan = Sine(freq=self.lfo * randRange(1.0, 2.0), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=[f, f*1.01]))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.98,.99], damp=.5, bal=reverb))
        # self.pipeline.append(Delay(self.pipeline[-1], delay=[.05, .06], feedback=.5))
        self.out = self.pipeline[-1]

        self.out = self.lfoMul * self.env * self.out
        self.out.out()

        self.pat = Pattern(self.play, dur)
        self.pat.play()

    def play(self):
        self.env.play()

# -----------------------------------------------------------------------------------------------

root = 25
dampFactor = 1.0
cutoff = 200

cutoff = Sig(20000)
phase = Sig(.75)
dt = Sig(.2)
reverb = Sig(.3)

cutoff.ctrl([SLMap(1., 20000., 'lin', "value", cutoff.value)], "Cutoff", ) # NOTE: the 'name' must be the name of attribute
phase.ctrl([SLMap(0., 1., 'lin', "value", phase.value)], "Phase", ) # NOTE: the 'name' must be the name of attribute
dt.ctrl([SLMap(0.1, 0.5, 'lin', "value", dt.value)], "Tempo") # NOTE: the 'name' must be the name of attribute
reverb.ctrl([SLMap(0.0, 1.0, 'lin', "value", reverb.value)], "Reverb") # NOTE: the 'name' must be the name of attribute

# Put wanted profile last.
muls = [0.623, 0.812, 0.396, 0.158, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.065, 0.000, 0.062, 0.092, 0.035, 0.000, 0.000] # Ambient.22.Dark.1.mp3
muls = [0.000, 0.000, 0.000, 0.000, 0.179, 0.056, 0.080, 0.030, 0.067, 0.019, 0.040, 0.010, 0.000, 0.000, 0.020, 0.010, 0.006] # AmbientE.Beat.High.1.mp3
muls = [0.800, 0.800, 0.326, 0.123, 0.179, 0.056, 0.080, 0.030, 0.067, 0.019, 0.040, 0.010, 0.040, 0.040, 0.025, 0.008, 0.000] # AmbientE.Beat.1.mp3

oscs = [
    # Pad(note=root     , bw=40, damp=dampFactor*0.9, cutoff=cutoff, reverb=reverb, mul=muls[ 0], phase=phase),
    # Pad(note=root+12  , bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=reverb, mul=muls[ 1], phase=phase),
    # Pad(note=root+24  , bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=reverb, mul=muls[ 2], phase=phase),
    # Pad(note=root+24+7, bw=40, damp=dampFactor*0.8, cutoff=cutoff, reverb=reverb, mul=muls[ 3], phase=phase),
    # Pad(note=root+36  , bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=reverb, mul=muls[ 4], phase=phase),
    # Pad(note=root+36+7, bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=reverb, mul=muls[ 5], phase=phase),
    # Pad(note=root+48  , bw=50, damp=dampFactor*0.6, cutoff=cutoff, reverb=reverb, mul=muls[ 6], phase=phase),
    # Pad(note=root+48+7, bw=50, damp=dampFactor*0.5, cutoff=cutoff, reverb=reverb, mul=muls[ 7], phase=phase),
    # Pad(note=root+60  , bw=50, damp=dampFactor*0.4, cutoff=cutoff, reverb=reverb, mul=muls[ 8], phase=phase),
    # Pad(note=root+60+7, bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=reverb, mul=muls[ 9], phase=phase),
    # Pad(note=root+72  , bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=reverb, mul=muls[10], phase=phase),
    # Pad(note=root+72+7, bw=50, damp=dampFactor*0.3, cutoff=cutoff, reverb=reverb, mul=muls[11], phase=phase),
    Sample(note=root     , cutoff=cutoff,                          reverb=reverb, mul=muls[ 0], phase=phase),
    Sample(note=root+12  , cutoff=cutoff,                          reverb=reverb, mul=muls[ 1], phase=phase),
    Sample(note=root+24  , cutoff=cutoff,                          reverb=reverb, mul=muls[ 2], phase=phase),
    Sample(note=root+24+7, cutoff=cutoff,                          reverb=reverb, mul=muls[ 3], phase=phase),
    Sample(note=root+36  , cutoff=cutoff,                          reverb=reverb, mul=muls[ 4], phase=phase),
    Sample(note=root+36+7, cutoff=cutoff,                          reverb=reverb, mul=muls[ 5], phase=phase),
    Sample(note=root+48  , cutoff=cutoff,                          reverb=reverb, mul=muls[ 6], phase=phase),
    Sample(note=root+48+7, cutoff=cutoff,                          reverb=reverb, mul=muls[ 7], phase=phase),
    Sample(note=root+60  , cutoff=cutoff,                          reverb=reverb, mul=muls[ 8], phase=phase),
    Sample(note=root+60+7, cutoff=cutoff,                          reverb=reverb, mul=muls[ 9], phase=phase),
    Sample(note=root+72  , cutoff=cutoff,                          reverb=reverb, mul=muls[10], phase=phase),
    Sample(note=root+72+7, cutoff=cutoff,                          reverb=reverb, mul=muls[11], phase=phase),
    ToneBeatSimple(note=root   , dur=dt, decay=.12, sustain=.0,    reverb=reverb, mul=muls[12], phase=phase, tone=[1., .8, .4, .2, .05]),
    ToneBeatSimple(note=root+12, dur=dt, decay=.12, sustain=.0,    reverb=reverb, mul=muls[13], phase=phase, tone=[1., .8, .4, .2, .05]),
    ToneBeatSimple(note=root+24, dur=dt, decay=.12, sustain=.0,    reverb=reverb, mul=muls[14], phase=phase, tone=[1., .8, .4, .2, .05]),
    ToneBeatSimple(note=root+36, dur=dt, decay=.12, sustain=.0,    reverb=reverb, mul=muls[15], phase=phase, tone=[1., .4, .2, .1, .05]),
    ToneBeatSimple(note=root+48, dur=dt, decay=.12, sustain=.0,    reverb=reverb, mul=muls[16], phase=phase, tone=[1., .4, .2, .1, .05]),
]

# -----------------------------------------------------------------------------------------------

def printValue(v):
    print("{:.3f}, ".format(v), end='')

def printInfos():
    [printValue(osc.amp.value.value) if hasattr(osc.amp.value, 'value') else printValue(osc.amp.value) for osc in oscs]
    print("")

pool = Pattern(printInfos, 3)
pool.play()

s.gui(locals())
