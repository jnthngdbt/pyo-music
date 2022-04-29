from pyo import *
import numpy as np

s = Server().boot()

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()

dur = 0.2

class PalmMute:
  def __init__(self, root=49, notes=[0,3,5,7,10], spec=[1.], dur=0.25, extend=False, mul=1.0) -> None:
    self.idx = 0
    self.root = root
    self.notes = notes

    self.mul = Sig(mul)
    self.mul.ctrl()

    self.env = Linseg([(0.0000,0.0000),(0.0115,0.5017),(0.0224,0.2237),(dur,0.0000)])
    self.env.graph()

    self.table = HarmTable(spec) #, 0.3, 0.1, 0.02, 0.005
    self.osc = Osc(table=self.table, freq=100, phase=[0, 0.5], mul=0.2*self.mul*self.env)
    self.osc.out()

    self.pat = Pattern(self.play, time=dur)
    self.pat.play()

  def play(self):
    note = self.root + self.notes[self.idx]
    f = midiToHz(note)
    self.osc.freq = f
    self.env.play()
    self.idx = (self.idx + 1) % len(self.notes)

base = 25
r = [
    PalmMute(root=base   , notes=[ 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7], dur=dur, mul=1, spec=[.8, 1, .8 , .5, .3, .2, .1, .05]), # , .5, .3, .2, .1, .05
    # PalmMute(root=base+12, notes=[12,12,12,11,11,11,12,12,19,19,19,12,12,12,11,11], dur=dur, mul=0.2, spec=[1, 1, .2]),
    PalmMute(root=base+12, notes=[16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,17,16,16,16,16,16,16,16,16,14,14,14,14,14,14,14,14], dur=dur, mul=0.1, spec=[.8, 1, .8]), # , .5, .3, .2, .1, .05
    PalmMute(root=base+24, notes=[12,7,11,7], dur=dur, mul=0.05, spec=[1,1]),
    # PalmMute(root=base+60, notes=[ 0, 5, 7,12], dur=dur, mul=0.005, spec=[1]),
]



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
