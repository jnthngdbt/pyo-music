from pyo import *
import numpy as np

s = Server().boot()

h = HarmTable([x**-2.0 for x in np.arange(1, 8)])

root = 40
lfo = 0.05
ampLevel = 0.1
ampScale = 0.5 * ampLevel

fa = midiToHz([0 + root ])
fb = midiToHz([7 + root ])
fc = midiToHz([12 + root ])
fd = midiToHz([16 + root ])

lfa = Sine(freq=lfo + 0.00, phase=0.2)
lfb = Sine(freq=lfo - 0.02, phase=0.4)
lfc = Sine(freq=lfo + 0.02, phase=0.6)
lfd = Sine(freq=lfo - 0.02, phase=0.8)

a = Osc(h, freq=fa, mul=1.0 * ampLevel + ampScale * lfa).mix(2)
b = Osc(h, freq=fb, mul=1.0 * ampLevel + ampScale * lfb).mix(2)
c = Osc(h, freq=fc, mul=0.8 * ampLevel + ampScale * lfc).mix(2)
d = Osc(h, freq=fd, mul=0.5 * ampLevel + ampScale * lfd).mix(2)

x = a + b + c + d
x.out()

s.gui(locals())