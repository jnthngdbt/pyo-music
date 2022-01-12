from pyo import *
import numpy as np

s = Server().boot()
s.start()

padSize = 262144
padFreq = 440
padRatio = s.getSamplingRate() / padSize

# For high note (85-97)
oscFreqHigh = padRatio * midiToHz(97) / padFreq
th = PadSynthTable(
  basefreq=padFreq, 
  size=padSize, 
  spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
  bw=40, # 20: org, 50: def strings, 70: dreamy more highs
  bwscl=1.5, # 1: def string, 2: dreamy, 3: flute/wind
  damp=0.2, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
  nharms=64, # 64: def 
)

# For low note (36)
# - for low string: damp=0.9 (grittier)
# - for low bass: damp=0.5
# - for low growl: midi 25, damp=0.9 (0.95 more open)
oscFreqLow = padRatio * midiToHz(25) / padFreq
tl = PadSynthTable(
  basefreq=padFreq, 
  size=padSize, 
  spread=1, # 1: def strings, 2:shallower/pure, in between: creepy (near 1, slight dissonnance) // freq = basefreq * pow(n, spread)
  bw=50, # 20: org, 50: def strings, 70: dreamy more highs
  bwscl=1, # 1: def string, 2: dreamy, 3: flute/wind
  damp=.95, # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
  nharms=64, # 64: def 
)

# # lfotl = TriangleTable() 
# # lfoth = TriangleTable() 
# # lfol = Osc(lfotl, freq=0.008, phase=0.75).range(0.1, 1.0)
# # lfoh = Osc(lfotl, freq=0.012, phase=0.75).range(0.1, 1.0)

# lfol = Sine(freq=0.02, phase=0.75).range(0.1, 1.0)
# lfoh = Sine(freq=0.03, phase=0.75).range(0.1, 1.0)

# xl = Osc(tl, freq=oscFreqLow, mul=lfol)
# xh = Osc(th, freq=oscFreqHigh, mul=lfoh)

xl = Osc(tl, freq=oscFreqLow)
xh = Osc(th, freq=oscFreqHigh)

x = xl + .1 * xh

d = Delay(x, delay=[.1,.2], feedback=.5, mul=.7)
d.out()

s.gui(locals())