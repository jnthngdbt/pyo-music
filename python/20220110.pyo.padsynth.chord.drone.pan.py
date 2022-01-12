from pyo import *
import numpy as np

s = Server().boot()
s.start()

#        0  1  2  3  4  5  6  7  8  9  10 11
scale = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0] # major
scale = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] # minor small
scale = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # minor
scale = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] # power
scale = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # peak
scale = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] # major small
scale = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0] # riff

root = 36
octaves = [0,1,2]

padSize = 262144
padFreq = 440
padRatio = s.getSamplingRate() / padSize

notes = np.nonzero(scale)[0]
for i in octaves:
  notes = np.concatenate([notes, i * 12 + notes], axis=0)

nbNotes = len(notes)

table = PadSynthTable(
  basefreq=padFreq, 
  size=padSize, 
  spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
  bw=50, # 20: org, 50: def strings, 70: dreamy more highs
  bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
  damp=0.7, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
  nharms=64, # 64: def 
)

def randRange(a,b):
  r = b - a
  return a + r * np.random.rand()   

oscLfos = [None] * nbNotes
oscs = [None] * nbNotes

panLfos = [None] * nbNotes
pans = [None] * nbNotes

for i, note in enumerate(notes):
  f = midiToHz(note + root)
  freqRatio = padRatio * f / padFreq

  oscLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=0.75).range(0, 1)
  oscs[i] = Osc(table, freq=freqRatio, mul=oscLfos[i])

  panLfos[i] = Sine(freq=randRange(0.01, 0.04), phase=np.random.rand()).range(0.3, 0.7)
  pans[i] = Pan(oscs[i], outs=2, pan=panLfos[i])

x = 0.0 * Sine()
for sig in pans:
  x = x + sig
# x.out()

d = Delay(x, delay=np.arange(0.15, 0.3, 0.05).tolist(), feedback=.5, mul=.5)
d.ctrl()
# d.out()

r = Freeverb(d, size=[.79,.8], damp=.9, bal=.3, mul=0.3)
# r.out()
r.ctrl()

lfoLpTable = TriangleTable()
lfoLp = Osc(lfoLpTable, freq=0.01, phase=0.25).range(4000, 6000)
lp = MoogLP(r, freq=lfoLp, res=0.2)
lp.out()

# NOTE: Scope does not work with LFO: wx._core.wxAssertionError: C++ assertion "x > double((-2147483647 - 1))...

s.gui(locals())