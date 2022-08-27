from pyo import *
import numpy as np

s = Server().boot()
s.start()

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()

padFreq = 440
padSize = 2**19
padRatio = s.getSamplingRate() / padSize

t = PadSynthTable(
  basefreq=padFreq, 
  size=padSize, 
  spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
  bw=50, # 20: org, 50: def strings, 70: dreamy more highs
  bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
  damp=.7, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
  nharms=64, # 64: def 
)

root = 49 # c#  
amp = [1.]
notes = [0]
octaves = [0]

freq = expand(octaves=octaves, notes=[x + root for x in notes])
freq = padRatio * np.array(midiToHz(freq)) / padFreq

o = Osc(t, freq=freq.tolist())
d = Delay(o, delay=[.00, .05], feedback=0)

mix = .6 * d

p = Pan(mix, outs=2, pan=0.5)
p.out()

s.gui(locals())