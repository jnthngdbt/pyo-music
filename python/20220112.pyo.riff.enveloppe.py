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
      bwscl=1.5, # def: 1, breathing
      damp=0.7) # def: 0.7, mellow/bright

  def freq(self, midi=48):
    f = midiToHz(midi)
    f = np.array(f)
    f = self.ratio * f / self.basefreq
    return f.tolist()

class Instrument:
  def __init__(self):
    self.maxchnls = 16 # seems that number of channels (notes) stays fixed from instantiation
    self.pad = Pad()
    self.osc = Osc(self.pad.table, freq=np.zeros(self.maxchnls).tolist()).stop()
    self.pan = Pan(self.osc, outs=2, pan=0.5, spread=0.5).stop()
    self.delay = Delay(self.pan, delay=[0.15, 0.2, 0.25, 0.3], feedback=0.5).stop()
    self.env = Fader().stop()

    self.mix = 0.1 * self.delay * self.env

  def play(self, notes=[48, 55], dur=8, delay=0):
    notes = self.expand(notes=notes)
    self.osc.freq = self.pad.freq(notes)

    self.env.dur = dur
    self.env.fadein = 0.5 * dur
    self.env.fadeout = 0.5 * dur

    self.osc.play(dur=dur, delay=delay)
    self.pan.play(dur=dur, delay=delay)
    self.delay.play(dur=dur, delay=delay)
    self.env.play(dur=dur, delay=delay)

    self.mix.out(dur=dur, delay=delay)

    return self

  def expand(self, notes=[48, 52, 55], octaves=[0,1,2]):
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
ins = [Instrument(), Instrument()]

chords = [
  [45, 48, 52], # Am
  [41, 45, 48], # F
  [43, 48, 52], # C invert
  [43, 47, 50], # G
  [45, 48, 52], # Am
  [41, 45, 48, 53], # F power
  [48, 52, 55], # C
  [43, 47, 50, 55], # G power
]

dur = 12
delay = 0.6 * dur # somehow overlap must be < 0.5

i = 0

def playChord():
  global i
  insIdx = i % len(ins)
  chordIdx = i % len(chords)
  ins[insIdx].play(notes=chords[chordIdx], dur=dur)
  i += 1

peaks = [
  Peak(note=72, mul=0.7),
  # Peak(note=84, mul=0.4),
]

p = Pattern(playChord, time=delay).play()

s.gui(locals())