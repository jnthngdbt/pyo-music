from pyo import *
import numpy as np

s = Server().boot()

# major = [0,2,4,5,7,9,11]
major = [0,2,4,7,9]
minor = [0,3,5,7,10]

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()

class Arpeggio:
  def __init__(self, root=49, notes=minor, bps=4.0, decay=0.25, mirror=False, mul=1.0) -> None:
    self.idx = 0
    self.root = root
    self.notes = notes

    if mirror:
      self.notes = self.notes + self.notes[-2:0:-1] # weird slicing: all except first and last, reversed

    self.table = HarmTable([1, 0.3, 0.1, 0.02, 0.005])
    self.osc = Osc(table=self.table, freq=[100,101])

    self.env = Adsr(attack=.012, decay=decay, sustain=.0, release=.0, dur=.2)
    self.env.ctrl()

    self.out = mul * self.osc * self.env
    self.out.out()

    self.pat = Pattern(self.play, 1./bps)
    self.pat.play()

  def play(self):
    note = self.root + self.notes[self.idx]
    f = midiToHz(note)
    self.osc.freq = [f, f+1]
    self.env.play()
    self.idx = (self.idx + 1) % len(self.notes)

a = Arpeggio(root=37, notes=expand(minor, octaves=[0,1]), bps=10, decay=0.2, mirror=True, mul=0.1)

s.start()
s.gui(locals())