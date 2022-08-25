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
    def __init__(self, lfo=0.015, phase=Sig(.75), mul=1.):
        self.amp = Sig(mul)
        self.phase = phase
        self.lfoMul = self.amp * Sine(freq=lfo * randRange(1.6, 2.5), phase=randRange(0, 1)).range(0, 1)

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

        self.out.out()

# -----------------------------------------------------------------------------------------------

root = 73

mx = 1.
mn = .8

phase = Sig(.75)
phase.ctrl([SLMap(0., 1., 'lin', "value", phase.value)], "Phase", ) # NOTE: the 'name' must be the name of attribute

oscs = [
    Peak(note=root-48+ 0, mul=mn, phase=phase),
    Peak(note=root-36+ 0, mul=mn, phase=phase),
    Peak(note=root-24+ 0, mul=mn, phase=phase),
    Peak(note=root-12+ 0, mul=mn, phase=phase),
    Peak(note=root   + 0, mul=mx, phase=phase),
    Peak(note=root+12+ 0, mul=mn, phase=phase),
    Peak(note=root-12+ 4, mul=mn, phase=phase),
    Peak(note=root   + 4, mul=mx, phase=phase),
    Peak(note=root+12+ 4, mul=mn, phase=phase),
    Peak(note=root-12+ 5, mul=mn, phase=phase),
    Peak(note=root   + 5, mul=mx, phase=phase),
    Peak(note=root+12+ 5, mul=mn, phase=phase),
    Peak(note=root-12+ 7, mul=mn, phase=phase),
    Peak(note=root   + 7, mul=mx, phase=phase),
    Peak(note=root+12+ 7, mul=mn, phase=phase),
    Peak(note=root-12+ 9, mul=mn, phase=phase),
    Peak(note=root   + 9, mul=mx, phase=phase),
    Peak(note=root+12+ 9, mul=mn, phase=phase),
]

drone = Sig(1.)
drone.ctrl([SLMap(0., 10., 'lin', "value", drone.value)], "Drone") # NOTE: the 'name' must be the name of attribute
d = Peak(note=root+12, mul=drone, lfo=0, phase=.25)

# -----------------------------------------------------------------------------------------------

s.gui(locals())
