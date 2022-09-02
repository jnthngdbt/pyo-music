from pyo import *
import numpy as np

s = Server().boot()
s.start()

class Pad:
  def __init__(self):
    self.size = 262144
    self.basefreq = 440
    self.ratio = s.getSamplingRate() / self.size
    self.table = PadSynthTable(basefreq=self.basefreq, size=self.size, 
      spread=1, # for slight dissonnance
      bw=50, # def: 50, pure > noisy
      bwscl=1.2, # def: 1, breathing
      damp=0.6) # def: 0.7, mellow/bright

  def freq(self, midi=48):
    f = midiToHz(midi)
    f = np.array(f)
    f = self.ratio * f / self.basefreq
    return f.tolist()

class Instrument:
  def __init__(self, mul=.2):
    self.maxchnls = 16 # seems that number of channels (notes) stays fixed from instantiation
    self.pad = Pad()
    self.env = Osc(HannTable(), freq=1).stop()
    self.osc = Osc(self.pad.table, freq=np.zeros(self.maxchnls).tolist(), mul=mul*self.env)
    self.pan = Pan(self.osc, outs=2, pan=.5) # merge voices to mono
    self.out = Delay(self.pan, delay=[.00, .1]) # stereo effect
    self.out.out()

  def play(self, notes, dur, octaves):
    notes = self.expand(notes=notes, octaves=octaves)
    self.osc.freq = self.pad.freq(notes)

    self.env.freq = 1./dur
    self.env.play(dur=dur)

    return self

  def expand(self, notes, octaves):
    x = []
    notes = np.array(notes)
    for i in octaves:
      x = np.concatenate([x, i * 12 + notes], axis=0)
    return x.tolist()

class Peak:
  def __init__(self, note=48, mul=1.0) -> None:
    self.noise1 = PinkNoise()
    self.noise2 = PinkNoise()
    self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=9, type=2, stages=2, mul=mul).out()

# Using 2 instances to allow overlapping.
mul = .1
ins = [Instrument(mul=mul), Instrument(mul=mul)]

root = 48
octaves = [0]

chords = [
  [x + root for x in [-3, 0, 4]], # Am
  [x + root for x in [-7, -3, 0, 5]], # F
  [x + root for x in [-5, 0, 4]], # C invert
  [x + root for x in [-5, -1, 2]], # G
  # [x + root for x in [-3, 0, 4]], # Am
  # [x + root for x in [-7, -3, 0, 5]], # F power
  # [x + root for x in [0, 4, 7]], # C
  # [x + root for x in [-5, -1, 2, 7]], # G power
]

dur = 12
overlap = 0.2
interval = (1.-overlap) * dur # somehow overlap must be < 0.5

i = 0

def playChord():
  global i
  insIdx = i % len(ins)
  chordIdx = i % len(chords)
  ins[insIdx].play(notes=chords[chordIdx], dur=dur, octaves=octaves)
  i += 1

peaks = [
  Peak(note=72, mul=0.6),
  # Peak(note=84, mul=0.4),
]

p = Pattern(playChord, time=interval).play()

s.gui(locals())