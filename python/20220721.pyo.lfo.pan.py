from pyo import *
import numpy as np

s = Server().boot()
s.start()

def randRange(a,b):
    r = b - a
    return a + r * np.random.rand()

class Pad:
    def __init__(self, note, bw, damp, lfo=0.015, cutoff=20000, mul=1.):

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

        self.lfoMul = Sine(freq=lfo * randRange(0.6, 1.5), phase=0.75).range(0, mul)
        self.lfoPan = Sine(freq=lfo * randRange(0.6, 1.5), phase=np.random.rand()).range(0.3, 0.7)

        self.pipeline = []
        self.pipeline.append(Osc(self.table, freq=freqRatio))
        self.pipeline.append(Pan(self.pipeline[-1], outs=2, pan=self.lfoPan, mul=self.lfoMul))
        self.pipeline.append(Delay(self.pipeline[-1], delay=np.arange(0.15, 0.3, 0.05).tolist(), feedback=.5))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.79,.8], damp=.9, bal=.3))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class Recorded:
    def __init__(self, soundPath, reverb=.3, lfo=.002, cutoff=20000, mul=1.) -> None:
        self.table = SndTable(soundPath)
        self.freq = self.table.getRate()

        self.lfoMul = Sine(freq=lfo * randRange(0.6, 1.5), phase=0.75).range(0, mul)

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=self.freq, mul=self.lfoMul))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.79,.8], damp=.9, bal=reverb))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()

class RecordedGuit:
    def __init__(self, soundPath, reverb=.3, lfo=.002, cutoff=20000, mul=1.) -> None:
        self.table = SndTable(soundPath)
        self.freq = self.table.getRate()

        self.lfoMul = Sine(freq=lfo * randRange(0.6, 1.5), phase=0.75).range(0, mul)

        self.pipeline = []
        self.pipeline.append(Osc(table=self.table, freq=self.freq, mul=self.lfoMul))
        self.pipeline.append(Freeverb(self.pipeline[-1], size=[.79,.8], damp=.9, bal=reverb))
        self.pipeline.append(Delay(self.pipeline[-1], delay=[.05, .06], feedback=.5))
        self.pipeline.append(MoogLP(self.pipeline[-1], freq=cutoff, res=0.0))
        self.out = self.pipeline[-1]

        self.out.out()


root = 27
dampFactor = 1.0
cutoff = 10000
volume = .4

oscs = [
    Pad(note=root     , bw=40, damp=dampFactor*0.9, cutoff=cutoff, mul=volume*.7),
    Pad(note=root+12  , bw=40, damp=dampFactor*0.9, cutoff=cutoff, mul=volume*.7),
    Pad(note=root+36  , bw=50, damp=dampFactor*0.5, cutoff=cutoff, mul=volume*.4),
    Pad(note=root+36+7, bw=50, damp=dampFactor*0.5, cutoff=cutoff, mul=volume*.4),
    Pad(note=root+60  , bw=50, damp=dampFactor*0.4, cutoff=cutoff, mul=volume*.15),
    # Recorded(soundPath="./data/Clean Combo#03.wav", cutoff=cutoff, mul=volume*4.0, reverb=0.),
    # Recorded(soundPath="./data/Clean Combo#01.wav", cutoff=cutoff, mul=volume*2.0, reverb=0.5),
    Recorded(soundPath="./data/Clean Combo#05.wav", cutoff=cutoff, mul=volume*2.0, reverb=0.5),
    RecordedGuit(soundPath="./data/AmbientE Loops.Guit.1.wav", cutoff=cutoff, mul=volume*0.08, lfo=.003, reverb=0.1),
    RecordedGuit(soundPath="./data/AmbientE Loops.Guit.3.wav", cutoff=cutoff, mul=volume*0.08, lfo=.003, reverb=0.1),
    RecordedGuit(soundPath="./data/AmbientE Loops.Guit.4.wav", cutoff=cutoff, mul=volume*0.08, lfo=.003, reverb=0.1),
]

s.gui(locals())
