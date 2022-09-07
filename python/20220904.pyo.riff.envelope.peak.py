from pyo import *
import numpy as np

s = Server().boot()
s.start()

class Peak:
  def __init__(self, note, mul) -> None:
    f = midiToHz(note)
    self.noise = BrownNoise()
    self.band = Biquadx(self.noise, freq=np.zeros(8).tolist(), q=2.3, type=2, stages=3, mul=.3*mul)
    self.peak = Biquadx(self.noise, freq=np.zeros(8).tolist(), q=26,  type=2, stages=2, mul=10.*mul)
    self.osc = self.band + self.peak

class InstrumentPeak:
  def __init__(self, mul):
    self.maxchnls = 16 # seems that number of channels (notes) stays fixed from instantiation
    self.env = Osc(HannTable(), freq=1).stop()
    self.peak = Peak(note=1, mul=mul*self.env)
    self.osc = self.peak.osc.mix(1) # merge voices to mono
    self.out = Delay(self.osc, delay=[.00, .1]) # stereo effect
    self.out.out()

  def expand(self, notes, octaves):
    x = []
    notes = np.array(notes)
    for i in octaves:
      x = np.concatenate([x, i * 12 + notes], axis=0)
    return x.tolist()

  def play(self, notes, dur, octaves):
    notes = self.expand(notes=notes, octaves=octaves)
    f = midiToHz(notes)
    self.peak.band.setFreq(f)
    self.peak.peak.setFreq(f)

    self.env.freq = 1./dur
    self.env.play(dur=dur)

    return self

  def stop(self):
    self.env.stop()
    return self

# Using 2 instances to allow overlapping.
mul = .6; ins = [InstrumentPeak(mul=mul), InstrumentPeak(mul=mul)]

root = 61
octaves = [0,1,2]

chords = [
  [x + root for x in [ 9, 12]], # Am
  [x + root for x in [ 5, 12]], # F
  [x + root for x in [12, 16]], # C invert
  [x + root for x in [ 7, 14]], # G
]

dur = 12
overlap = 0.3 # can not be >=0.5, seems that cannot play() if not finished
interval = (1.-overlap) * dur # somehow overlap must be < 0.5

i = 0

def playChord():
  global i
  insIdx = i % len(ins)
  chordIdx = i % len(chords)
  ins[insIdx].stop()
  ins[insIdx].play(notes=chords[chordIdx], dur=dur, octaves=octaves)
  i += 1

p = Pattern(playChord, time=interval).play()

S = Spectrum(ins[0].out)

s.gui(locals())