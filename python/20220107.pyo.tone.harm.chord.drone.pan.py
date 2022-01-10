from pyo import *
import numpy as np

s = Server().boot()
s.start()

h = HarmTable([x**-1.8 for x in np.arange(1, 10)])
# h = HarmTable([.1,0,.2,0,.1,0,0,0,.04,0,0,0,0.02])
# h = HarmTable([.5, .7, .7, .5, .3, .1, .05, .02])

root = 40
notes = [0, 4, 7, 9, 12]
nbNotes = len(notes)

def randRange(a,b):
  r = b - a
  return a + r * np.random.rand()

def randPhase():
  return 2.0 * np.pi * np.random.rand()

oscLfos = [None] * nbNotes
oscs = [None] * nbNotes

panLfos = [None] * nbNotes
pans = [None] * nbNotes

for i, note in enumerate(notes):
  f = midiToHz(note + root)

  oscLfos[i] = LFO(randRange(0.01, 0.04), sharp=0.5, type=3).range(0, 1)
  oscs[i] = Osc(h, freq=[f, f + 1], mul=oscLfos[i]).mix(2)

  panLfos[i] = LFO(randRange(0.01, 0.04), sharp=0.5, type=3).range(0.3, 0.7)
  pans[i] = Pan(oscs[i], outs=2, pan=panLfos[i])

x = 0.0 * Sine()
for sig in pans:
  x = x + sig

d = Delay(x, delay=[.15,.2,.25, .3, .35], feedback=.5, mul=.4)
# d.out()
d.ctrl()

r = Freeverb(d, size=[.79,.8], damp=.9, bal=.3, mul=0.3)
r.out()
r.ctrl()

# NOTE: Scope does not work with LFO: wx._core.wxAssertionError: C++ assertion "x > double((-2147483647 - 1))...

s.gui(locals())