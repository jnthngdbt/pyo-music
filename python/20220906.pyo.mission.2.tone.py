
from pyo import *

s = Server().boot()
s.start()

class Horn:
  def __init__(self, f, tone, mul=1.) -> None:
    self.p = []
    self.p.append(HarmTable(tone)) # sub 0, main, fifth, octave
    self.p.append(Osc(self.p[-1], f))
    self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0)) # adds some movement
    self.p.append(Freeverb(self.p[-1], .8)) # adds some movement
    self.p[-1].mul = mul
    self.p[-1].out()

class BassString:
  def __init__(self, f, mul) -> None:
    padSize = 262144 * 2
    padFreq = 440
    padRatio = s.getSamplingRate() / padSize
    freqRatio = padRatio * f / padFreq
    self.p = []
    self.p.append(Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=50, bwscl=1.05, damp=.7, nharms=64), freq=freqRatio))
    self.p.append(EQ(self.p[-1], f, q=10, boost=-40)) # adds some movement
    # self.p.append(EQ(self.p[-1], f*2, q=10, boost=-10)) # adds some movement
    self.p[-1].ctrl()
    self.p.append(Freeverb(self.p[-1], [.78, .80])) # adds some movement
    self.p[-1].mul = mul
    self.p[-1].out()

class Baseline:
  def __init__(self, mul) -> None:
    self.p = []
    self.p.append(BrownNoise())
    self.p.append(Delay(self.p[-1], delay=[0, .1])) # adds some movement
    self.p[-1].mul = mul
    self.p[-1].out()

x = [
  # Horn(40, mul=.08, tone=[1, .15, .15, .2]), # horn
  BassString(52, mul=1), # horn
  Horn(310, mul=.008, tone=[1., .7, .25]), # horn
  # MissionTwoTone(261.35, mul=.05, tone=[1.,.5, .28, .08]), # horn
  Baseline(mul=.01),
]



Spectrum([xi.p[-1] for xi in x], size=8192)

s.gui(locals())