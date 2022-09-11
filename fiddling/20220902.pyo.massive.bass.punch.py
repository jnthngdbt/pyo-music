
from pyo import *

s = Server().boot()
s.start()

class BassString:
  def __init__(self, f) -> None:
    padSize = 262144 * 2
    padFreq = 440
    padRatio = s.getSamplingRate() / padSize
    freqRatio = padRatio * f / padFreq
    self.osc = Osc(PadSynthTable(basefreq=padFreq, size=padSize, spread=1, bw=50, bwscl=1.0, damp=.9, nharms=64), freq=freqRatio)

  def out(self): 
    self.osc.out()

class BassTone:
  def __init__(self, f) -> None:
    self.table = HarmTable([.5, 1, .8, .5])
    self.osc = Osc(self.table, f)

  def out(self): 
    self.osc.out()

class BigBass:
  def __init__(self, instrument, mul=1, dur=1) -> None:
    self.env = Adsr(attack=.01, decay=.3, sustain=.8, release=.15, dur=dur)
    self.env.setExp(1)
    self.env.ctrl()

    self.p = []
    self.p.append(instrument)
    self.p.append(Freeverb(self.p[-1].osc, size=[.7,.8], bal=0.5)) # adds some movement
    self.p[-1].mul = mul * self.env
    self.p[-1].out()

  def play(self):
    self.env.play()

dur = 2.5
interval = 4

B = [
  BigBass(instrument=BassTone(midiToHz(25)), mul=.3, dur=dur),
  BigBass(instrument=BassString(midiToHz(25)), mul=.5, dur=dur),
  BigBass(instrument=BassString(midiToHz(37)), mul=.3, dur=dur),
]

def playEnv():
  [b.play() for b in B]

p = Pattern(playEnv, time=interval).play()

s.gui(locals())