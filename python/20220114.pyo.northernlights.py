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

# -----------------------------------------------------------
# padFreq = 440
# padSize = 2**19
# padRatio = s.getSamplingRate() / padSize
# t = PadSynthTable(basefreq=padFreq, size=padSize, damp=0.7, bw=50, bwscl=1, spread=1)

# root = 49 # c#
# intervals = [0, 9, 16]
# amp = [1.,.6, .1]
# octaves = [0]

# notes = [x + root for x in intervals]
# # freq = expand(octaves=octaves, notes=)
# freq = padRatio * np.array(midiToHz(notes)) / padFreq

# o = Osc(t, freq=freq.tolist(), mul=amp)
# d = Delay(o, delay=[.00, .05], feedback=0)
# # c = Chorus(d, depth=1, feedback=0.25, bal=0.5)
# # c.ctrl()
# b = Biquadx(d, freq=midiToHz(85), q=1, type=2, stages=2)

# mix = 4. * b

# p = Pan(mix, outs=2, pan=.5, spread=0)
# p.out()
# -----------------------------------------------------------

root = 85 # c#
amp = [1., .5, .4]
notes = [0, 5, 7]

freq = expand(octaves=[0], notes=[x + root for x in notes])

noise1 = PinkNoise()
noise2 = PinkNoise()
b = Biquadx([noise1, noise2], freq=midiToHz(freq), mul=amp, q=40, type=2, stages=2)
p = Pan(b, outs=2, pan=0.5)

mix = 2.0 * p
mix.out()

# -----------------------------------------------------------

# Peaks, expanded, bandpass


# class Peak:
#   def __init__(self, note=48, mul=1.0) -> None:
#     q = 30 # 10 + 0.3*note
#     self.noise1 = PinkNoise()
#     self.noise2 = PinkNoise()
#     self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=q, type=2, stages=2, mul=mul)
#     # self.band.ctrl()

#   def out(self):
#     self.band.out()

# root = 49 # c#
# intervals = [0, 5]
# octaves = [0,1,2,3,4]

# notes = [x + root for x in intervals]
# notes = expand(notes, octaves)

# # peaks = [
# #   # Peak(note=66, mul=0.5),
# #   Peak(note=73, mul=0.4),
# #   Peak(note=78, mul=0.6),
# #   Peak(note=85, mul=0.3),
# #   Peak(note=92, mul=0.2),
# #   # Peak(note=95, mul=0.2),
# # ]

# peaks = [Peak(n) for n in notes]
# pk = [peak.band for peak in peaks]

# p = Pan(pk, outs=2, pan=0.5)

# # p.out()

# # lfoLpTable = TriangleTable()
# # lfoLp = Osc(lfoLpTable, freq=.05, phase=0.25).range(1000, 4000)
# # lp = MoogLP(p, freq=lfoLp, res=1)
# # # Reson
# # lp.out()

# rs = Biquadx(p, freq=2000, q=10, stages=2, type=2)
# # rs = Resonx(p, freq=2000, q=10, stages=2)
# rs.out()
# rs.ctrl()

# -----------------------------------------------------------

s.gui(locals())