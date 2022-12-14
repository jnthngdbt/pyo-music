from pyo import *
import numpy as np

s = Server().boot()

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()

notes = [0,2,4,5,7,9,11] # major
notes = [0,2,4,7,9] # major
notes = [0,7,10] # minor
notes = [0,3,5,7,10] # minor
notes = [0,4,7,11] # major

class Arpeggio:
  def __init__(self, root=49, notes=[0,3,5,7,10], time=0.2, dur=0.2, sustain=0.75, doMirror=False, mul=1.0) -> None:
    self.idx = 0
    self.root = root
    self.notes = notes

    if doMirror:
      self.notes = self.notes + self.notes[-2:0:-1] # weird slicing: all except first and last, reversed

    self.table = HarmTable([1]) # , 0.3, 0.1, 0.02, 0.005
    self.osc = Osc(table=self.table, freq=[100,101])

    self.effect = Freeverb(self.osc, size=0.8, damp=0.5, bal=0.5)
    self.effect.ctrl()

    p = 0.012
    self.env = Adsr(attack=p, decay=p, sustain=sustain, release=p, dur=dur)
    self.env.ctrl()

    self.out = mul * self.effect * self.env
    self.out.out()

    self.pat = Pattern(self.play, time)
    self.pat.play()

  def play(self):
    note = self.root + self.notes[self.idx]
    f = midiToHz(note)
    self.osc.freq = [f, f+1]
    self.env.play()
    self.idx = (self.idx + 1) % len(self.notes)

t = 0.125
a = Arpeggio(root=73, notes=expand(notes, octaves=[0,1,2,3]), time=t, dur=t, sustain=.75, doMirror=False, mul=0.04)

s.start()
s.gui(locals())
