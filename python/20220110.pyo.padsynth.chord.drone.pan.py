from numpy.core.fromnumeric import size
from pyo import *
import numpy as np

s = Server().boot()
s.start()

# notes = [-2, 0, 3, 5, 7, 10, 12, 15, 17, 22, 24] # minor
notes = [0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24] # major

root = 20
nbNotes = len(notes)

padSize = 262144
padFreq = 440
padRatio = s.getSamplingRate() / padSize

table = PadSynthTable(basefreq=padFreq, size=padSize)

def randRange(a,b):
  r = b - a
  return a + r * np.random.rand()   

oscLfos = [None] * nbNotes
oscs = [None] * nbNotes

panLfos = [None] * nbNotes
pans = [None] * nbNotes

for i, note in enumerate(notes):
  f = midiToHz(note + root)

  freqRatio = f / padFreq

  oscLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=np.random.rand()).range(0, 1)
  oscs[i] = Osc(table, freq=freqRatio, mul=oscLfos[i]).mix(2)

  panLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=np.random.rand()).range(0.3, 0.7)
  pans[i] = Pan(oscs[i], outs=2, pan=panLfos[i])

x = 0.0 * Sine()
for sig in pans:
  x = x + sig

d = Delay(x, delay=[.15,.2,.25, .3, .35], feedback=.5, mul=.4)
d.ctrl()

r = Freeverb(d, size=[.79,.8], damp=.9, bal=.3, mul=0.3)
r.out()
r.ctrl()

# NOTE: Scope does not work with LFO: wx._core.wxAssertionError: C++ assertion "x > double((-2147483647 - 1))...

s.gui(locals())