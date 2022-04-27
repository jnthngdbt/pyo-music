from pyo import *
import numpy as np

s = Server().boot()

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()

class PalmMute:
  def __init__(self, root=49, notes=[0,3,5,7,10], dur=0.25, decay=0.25, mul=1.0) -> None:
    self.idx = 0
    self.root = root
    self.notes = notes

    self.table = HarmTable([1, 0.3, 0.1, 0.02, 0.005])
    self.osc = Osc(table=self.table, freq=[100,101], mul=mul)

    self.env = Linseg([(0.0000,0.0000),(0.0047,0.8),(0.0109,0.47),(0.0943,0.0000)])
    self.env.graph()

    self.effect = Freeverb(self.osc, size=0.8, damp=0.5, bal=0.1, mul=self.env)
    self.effect.ctrl()
    self.effect.out()

    self.pat = Pattern(self.play, dur)
    self.pat.play()

  def play(self):
    note = self.root + self.notes[self.idx]
    f = midiToHz(note)
    self.osc.freq = [f, f+1]
    self.env.play()
    self.idx = (self.idx + 1) % len(self.notes)

dur = 0.2
r1 = PalmMute(root=37, notes=[ 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7], dur=dur, mul=0.5)
r2 = PalmMute(root=49, notes=[12,12,12,11,11,11,12,12,19,19,19,12,12,12,11,11], dur=dur, mul=0.1)
r3 = PalmMute(root=61, notes=[12], dur=2*dur, mul=0.05)

s.start()
s.gui(locals())













# from pyo import *
# import numpy as np
#
# s = Server().boot()
#
# notes = [0,2,4,5,7,9,11] # major
# notes = [0,2,4,7,9] # major
# notes = [0,3,5,7,10] # minor
# notes = [0,7,10] # minor
# notes = [0,7,11] # major
#
# def expand(notes=[48, 52, 55], octaves=[0,1,2]):
#   x = []
#   notes = np.array(notes)
#   for i in octaves:
#     x = np.concatenate([x, i * 12 + notes], axis=0)
#   return x.tolist()
#
# class Arpeggio:
#   def __init__(self, root=49, notes=[0,3,5,7,10], bps=4.0, decay=0.25, mirror=False, mul=1.0) -> None:
#     self.idx = 0
#     self.root = root
#     self.notes = notes
#     self.dur = 1./bps
#
#     if mirror:
#       self.notes = self.notes + self.notes[-2:0:-1] # weird slicing: all except first and last, reversed
#
#     self.env = CosTable([(0,0.0000),(393,0.5697),(965,0.2303),(3791,0.0000),(8191,0.0000)])
#     self.env.graph()
#
#     self.metro = Metro(self.dur).play()
#     self.amp = TrigEnv(self.metro, table=self.env, dur=.25, mul=mul)
#
#     self.idx = Count(self.metro, 0, len(notes))
#
#     self.table = HarmTable([1]) # , 0.3, 0.1, 0.02, 0.005
#     self.osc = Osc(table=self.table, freq=[0,0], mul=self.amp)
#
#     self.note = self.root + self.notes[self.idx]
#     self.freq = midiToHz(self.note)
#     self.osc.freq = [f, f+1]
#
#     # self.delay = Delay(self.osc, delay=np.arange(0,.04,.005).tolist(), feedback=0.5)
#     self.effect = Freeverb(self.osc, size=0.8, damp=0.5, bal=0.0)
#
#     self.effect.out()
#
# r1 = Arpeggio(root=37, notes=[9,9,9,9,9,9,9,9,5,5,5,5,5,5,5,5,0,0,0,0,0,0,0,0,7,7,7,7,7,7,7,7], bps=6, decay=0.1, mirror=False, mul=1)
#
# s.start()
# s.gui(locals())
