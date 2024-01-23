from pyo import *
import numpy as np

s = Server().boot()
s.start()

period = Sig(128)
period.ctrl([SLMap(0.1, 512, 'lin', "value", period.value)], "Period (s)", ) # NOTE: the 'name' must be the name of attribute

# To have sine: type=7, sharp=0, mul=0.5, add=0.5
# To smooth bottom peak ups: type=4, sharp=0, mul=1, add=0
lfo = LFO(freq=1./period, type=4, sharp=0, mul=1, add=0) # 0. Saw up (default) 1. Saw down 2. Square 3. Triangle 4. Pulse 5. Bipolar pulse 6. Sample and hold 7. Modulated Sine
lfo.ctrl()

freq = 300 + lfo * 8000

p = []
p.append([BrownNoise(), BrownNoise()])
p.append(ButLP(p[-1], freq=freq, mul=0.2))
p[-1].out()

Spectrum(p[-1])
Scope(lfo)

s.gui(locals())
