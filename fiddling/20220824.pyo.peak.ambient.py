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

# Stay below midi(30) or 60Hz for some growl.
class BigBass (Base):
    def __init__(self, note=25, **kwargs):
        super().__init__(**kwargs)

        self.p = []
        self.p.append(HarmTable([.5, 1, .2])) # sub 0, main, fifth, octave
        self.p.append(Osc(self.p[-1], midiToHz(note)))
        self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0, bal=.3)) # adds some movement
        self.p[-1].mul = self.lfoMul
        
        self.out = self.p[-1]

# -----------------------------------------------------------------------------------------------

root = 73

m0 = 1.
m1 = .8
m2 = .5

lfoMain = 0.03
lfoBass = 0.01

mulp = Sig(1.5)
mulp.ctrl([SLMap(0., 4., 'lin', "value", mulp.value)], "Peak volume") # NOTE: the 'name' must be the name of attribute

oscs = [
    BigBass(note=root-48+ 0, mul=.15*m0, lfo=lfoBass), # better if around midi(25)
    # BigBass(note=root-36+ 0, mul=.075*m1, lfo=lfoBass), # better if around midi(37)
    #
    Peak(note=root-24+ 0, mul=m2, mulp=mulp, lfo=lfoMain),
    Peak(note=root-12+ 0, mul=m1, mulp=mulp, lfo=lfoMain),
    Peak(note=root   + 0, mul=m0, mulp=mulp, lfo=lfoMain),
    Peak(note=root+12+ 0, mul=m1, mulp=mulp, lfo=lfoMain),
    # Peak(note=root+24+ 0, mul=m2, mulp=mulp, lfo=lfoMain),
    #
    Peak(note=root-24+ 4, mul=m2, mulp=mulp, lfo=lfoMain),
    Peak(note=root-12+ 4, mul=m1, mulp=mulp, lfo=lfoMain),
    Peak(note=root   + 4, mul=m0, mulp=mulp, lfo=lfoMain),
    Peak(note=root+12+ 4, mul=m1, mulp=mulp, lfo=lfoMain),
    # Peak(note=root+24+ 4, mul=m2, mulp=mulp, lfo=lfoMain),
    #
    Peak(note=root-24+ 5, mul=m2, mulp=mulp, lfo=lfoMain),
    Peak(note=root-12+ 5, mul=m1, mulp=mulp, lfo=lfoMain),
    Peak(note=root   + 5, mul=m0, mulp=mulp, lfo=lfoMain),
    Peak(note=root+12+ 5, mul=m1, mulp=mulp, lfo=lfoMain),
    # Peak(note=root+24+ 5, mul=m2, mulp=mulp, lfo=lfoMain),
    #
    Peak(note=root-24+ 7, mul=m2, mulp=mulp, lfo=lfoMain),
    Peak(note=root-12+ 7, mul=m1, mulp=mulp, lfo=lfoMain),
    Peak(note=root   + 7, mul=m0, mulp=mulp, lfo=lfoMain),
    Peak(note=root+12+ 7, mul=m1, mulp=mulp, lfo=lfoMain),
    #Peak(note=root+24+ 7, mul=m2, mulp=mulp, lfo=lfoMain),
    #
    Peak(note=root-24+ 9, mul=m2, mulp=mulp, lfo=lfoMain),
    Peak(note=root-12+ 9, mul=m1, mulp=mulp, lfo=lfoMain),
    Peak(note=root   + 9, mul=m0, mulp=mulp, lfo=lfoMain),
    Peak(note=root+12+ 9, mul=m1, mulp=mulp, lfo=lfoMain),
    ##Peak(note=root+24+ 9, mul=m2, mulp=mulp),
]

p = []
p.append(Mix([osc.out for osc in oscs], 2))
p.append(MoogLP(p[-1], freq=20000, res=0)); p[-1].ctrl()
# p.append(MoogLP(p[-1], freq=Osc(TriangleTable(), freq=0.005, phase=.9).range(1000, 5000), res=.1)); # LFO controlled cutoff
p[-1].out()

Spectrum(p[-1])

# -----------------------------------------------------------------------------------------------

s.gui(locals())
