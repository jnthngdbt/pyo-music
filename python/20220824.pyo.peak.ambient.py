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
    def __init__(self, note, mulp=2., mulb=.22, **kwargs):
        super().__init__(**kwargs)

        self.qp = 26
        self.qb = 2.3
        self.note = note
        self.freq = midiToHz(self.note)

        self.noise1 = BrownNoise()
        self.noise2 = BrownNoise()

        self.band = Biquadx([self.noise1, self.noise2], freq=self.freq, q=self.qb, type=2, stages=3, mul=mulb*self.lfoMul)
        self.peak = Biquadx([self.noise1, self.noise2], freq=self.freq, q=self.qp,  type=2, stages=2, mul=mulp*self.lfoMul)
        self.out = self.band + self.peak

# -----------------------------------------------------------------------------------------------

root = 73

m0 = 1.
m1 = .8
m2 = .5

mulp = Sig(2.)
mulp.ctrl([SLMap(0., 4., 'lin', "value", mulp.value)], "Peak volume") # NOTE: the 'name' must be the name of attribute

oscs = [
    Peak(note=root-48+ 0, mul=m1, mulp=mulp, lfo=0.01),
    Peak(note=root-36+ 0, mul=m1, mulp=mulp, lfo=0.01),
    #
    Peak(note=root-24+ 0, mul=m2, mulp=mulp),
    Peak(note=root-12+ 0, mul=m1, mulp=mulp),
    Peak(note=root   + 0, mul=m0, mulp=mulp),
    Peak(note=root+12+ 0, mul=m1, mulp=mulp),
    Peak(note=root+24+ 0, mul=m2, mulp=mulp),
    #
    Peak(note=root-24+ 4, mul=m2, mulp=mulp),
    Peak(note=root-12+ 4, mul=m1, mulp=mulp),
    Peak(note=root   + 4, mul=m0, mulp=mulp),
    Peak(note=root+12+ 4, mul=m1, mulp=mulp),
    Peak(note=root+24+ 4, mul=m2, mulp=mulp),
    #
    Peak(note=root-24+ 5, mul=m2, mulp=mulp),
    Peak(note=root-12+ 5, mul=m1, mulp=mulp),
    Peak(note=root   + 5, mul=m0, mulp=mulp),
    Peak(note=root+12+ 5, mul=m1, mulp=mulp),
    Peak(note=root+24+ 5, mul=m2, mulp=mulp),
    #
    Peak(note=root-24+ 7, mul=m2, mulp=mulp),
    Peak(note=root-12+ 7, mul=m1, mulp=mulp),
    Peak(note=root   + 7, mul=m0, mulp=mulp),
    Peak(note=root+12+ 7, mul=m1, mulp=mulp),
    Peak(note=root+24+ 7, mul=m2, mulp=mulp),
    #
    Peak(note=root-24+ 9, mul=m2, mulp=mulp),
    Peak(note=root-12+ 9, mul=m1, mulp=mulp),
    Peak(note=root   + 9, mul=m0, mulp=mulp),
    Peak(note=root+12+ 9, mul=m1, mulp=mulp),
    Peak(note=root+24+ 9, mul=m2, mulp=mulp),
]

p = []
p.append(Mix([osc.out for osc in oscs], 2))
p.append(MoogLP(p[-1], freq=20000, res=0)); p[-1].ctrl()
# p.append(MoogLP(p[-1], freq=Osc(TriangleTable(), freq=0.005, phase=.9).range(1000, 5000), res=.1)); # LFO controlled cutoff
p[-1].out()

Spectrum(p[-1])

# -----------------------------------------------------------------------------------------------

s.gui(locals())
