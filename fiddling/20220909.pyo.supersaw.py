from pyo import *
import numpy as np

s = Server().boot()
s.start()

f = midiToHz(37)

lfof = np.arange(.05, .15, 0.01).tolist()
lfo = Sine(lfof, phase=np.random.rand(len(lfof)).tolist()).range(1.2*f, 20*f)

p = []
p.append(SuperSaw(f, detune=.6, bal=.9)), p[-1].ctrl()
p.append(MoogLP(p[-1], freq=lfo, res=.9).mix(1)), p[-1].ctrl()
p.append(Delay(p[-1], delay=[.01, .21], feedback=.0, mul=.05)), p[-1].ctrl()
p[-1].out()

Spectrum(p[-1])

s.gui(locals())
