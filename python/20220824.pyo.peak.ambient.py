from attr import has
from pyo import *
import numpy as np

s = Server().boot()
s.start()

def randRange(a,b):
    r = b - a
    return a + r * np.random.rand()

# Inspired by (open in different tabs):
# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=00000030403000000000&a=2&am=5 (C#)
# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=00000030403000000000&a=2&am=5&d=4 (F)
# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=00000030403000000000&a=2&am=5&d=5 (F#)
# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=00000000304030000000&a=2&am=5&d=-5 (G#)
# https://mynoise.net/NoiseMachines/peakNoiseGenerator.php?l=00000000304030000000&a=2&am=5&d=-3 (A#)

class Base:
    def __init__(self, lfo=0.03, mul=1.):
        self.amp = Sig(mul)
        self.lfoMulFreq = Sine(freq=lfo*randRange(.2, .5), phase=randRange(0, 1)).range(lfo*.5, lfo*1.5) # LFO of amplitude LFO frequency
        self.lfoMul = self.amp * Sine(freq=self.lfoMulFreq, phase=randRange(0, 1)).range(0, 1) # LFO for amplitude modulation

class Peak (Base):
    def __init__(self, note, qp=26, qb=2.3, **kwargs):
        super().__init__(**kwargs)

        self.qp = qp
        self.qb = qb
        self.note = note
        self.freq = midiToHz(self.note)

        self.noise1 = BrownNoise()
        self.noise2 = BrownNoise()

        self.band = Biquadx([self.noise1, self.noise2], freq=self.freq, q=self.qb, type=2, stages=3, mul=.22*self.lfoMul)
        self.peak = Biquadx([self.noise1, self.noise2], freq=self.freq, q=self.qp,  type=2, stages=2, mul=2.*self.lfoMul)
        self.out = self.band + self.peak

# -----------------------------------------------------------------------------------------------

root = 73

m0 = 1.
m1 = .8
m2 = .5

oscs = [
    Peak(note=root-48+ 0, mul=m1),
    Peak(note=root-36+ 0, mul=m1),
    #
    Peak(note=root-24+ 0, mul=m2),
    Peak(note=root-12+ 0, mul=m1),
    Peak(note=root   + 0, mul=m0),
    Peak(note=root+12+ 0, mul=m1),
    Peak(note=root+24+ 0, mul=m2),
    #
    Peak(note=root-24+ 4, mul=m2),
    Peak(note=root-12+ 4, mul=m1),
    Peak(note=root   + 4, mul=m0),
    Peak(note=root+12+ 4, mul=m1),
    Peak(note=root+24+ 4, mul=m2),
    #
    Peak(note=root-24+ 5, mul=m2),
    Peak(note=root-12+ 5, mul=m1),
    Peak(note=root   + 5, mul=m0),
    Peak(note=root+12+ 5, mul=m1),
    Peak(note=root+24+ 5, mul=m2),
    #
    Peak(note=root-24+ 7, mul=m2),
    Peak(note=root-12+ 7, mul=m1),
    Peak(note=root   + 7, mul=m0),
    Peak(note=root+12+ 7, mul=m1),
    Peak(note=root+24+ 7, mul=m2),
    #
    Peak(note=root-24+ 9, mul=m2),
    Peak(note=root-12+ 9, mul=m1),
    Peak(note=root   + 9, mul=m0),
    Peak(note=root+12+ 9, mul=m1),
    Peak(note=root+24+ 9, mul=m2),
]

p = []
p.append(Mix([osc.out for osc in oscs], 2))
p.append(MoogLP(p[-1], freq=20000, res=.2)); p[-1].ctrl()
p[-1].out()

Spectrum(p[-1])

# -----------------------------------------------------------------------------------------------

s.gui(locals())
