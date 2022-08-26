from pyo import *
import numpy as np

s = Server().boot()
s.start()

def getSigVal(sig):
  return sig.value.value if hasattr(sig.value, 'value') else sig.value

class PeakSynth:
    def __init__(self, freq, qFactor, damp):
        self.noise1 = BrownNoise()
        self.noise2 = BrownNoise()

        Nh = 16

        f = [i * freq for i in range(1,Nh)]
        mb = [.22*Exp(-i*damp) for i in range(1,Nh)]
        mp = [2.*Exp(-i*damp) for i in range(1,Nh)]

        self.band = Biquadx([self.noise1, self.noise2], freq=f, q=qFactor*2.3, type=2, stages=3, mul=mb)
        self.peak = Biquadx([self.noise1, self.noise2], freq=f, q=qFactor*26,  type=2, stages=2, mul=mp)
        self.out = self.band + self.peak

# -----------------------------------------------------------------------------------------------

root = 61

qFactor = Sig(1.0)
qFactor.ctrl([SLMap(0.01, 2., 'lin', "value", qFactor.value)], "Qf", ) # NOTE: the 'name' must be the name of attribute

damp = Sig(.1)
damp.ctrl([SLMap(0.01, 1., 'lin', "value", damp.value)], "Damp", ) # NOTE: the 'name' must be the name of attribute

oscs = [
    PeakSynth(midiToHz(root), qFactor, damp)
]

p = []
p.append(Mix([osc.out for osc in oscs], 1, mul=1.))
p.append(Delay(p[-1], delay=[.00,.20]))
p[-1].out()

Spectrum(p[-1])

# -----------------------------------------------------------------------------------------------

s.gui(locals())
