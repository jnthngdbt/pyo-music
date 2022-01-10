from pyo import *
import numpy as np

s = Server().boot()

h = HarmTable([x**-1.8 for x in np.arange(1, 10)])
# h = HarmTable([.1,0,.2,0,.1,0,0,0,.04,0,0,0,0.02])
# h = HarmTable([.5, .7, .7, .5, .3, .1, .05, .02])

root = 40
lfo = 0.05
ampLevel = 0.1
ampScale = 0.5 * ampLevel

fa = midiToHz(0 + root )
fb = midiToHz(7 + root )
fc = midiToHz(12 + root)
fd = midiToHz(16 + root)

lfa = Sine(freq=lfo + 0.00, phase=0.2)
lfb = Sine(freq=lfo - 0.02, phase=0.4)
lfc = Sine(freq=lfo + 0.02, phase=0.6)
lfd = Sine(freq=lfo - 0.02, phase=0.8)

a = Osc(h, freq=[fa, fa + 1], mul=1.0 * ampLevel + ampScale * lfa).mix(2)
b = Osc(h, freq=[fb, fb + 1], mul=1.0 * ampLevel + ampScale * lfb).mix(2)
c = Osc(h, freq=[fc, fc + 1], mul=0.8 * ampLevel + ampScale * lfc).mix(2)
d = Osc(h, freq=[fd, fd + 1], mul=0.5 * ampLevel + ampScale * lfd).mix(2)

# Using amplitude LFOs for panning, but mixing sources.
pa = Pan(a, outs=2, pan=0.5 + 0.2 * lfd)
pb = Pan(b, outs=2, pan=0.5 + 0.2 * lfc)
pc = Pan(c, outs=2, pan=0.5 + 0.2 * lfb)
pd = Pan(d, outs=2, pan=0.5 + 0.2 * lfa)

x = pa + pb + pc + pd
# x = a + b + c + d
# x.out()

# lfp = Sine(freq=1, mul=.5, add=.5)
# p = Pan(x, outs=2, pan=lfp).out()

d = Delay(x, delay=[.15,.2,.25, .3, .35], feedback=.5, mul=.4)
d.out()
d.ctrl()

r = Freeverb(d, size=[.79,.8], damp=.9, bal=.3)
r.out()
r.ctrl()

# y = WGVerb(x).out()

s.gui(locals())