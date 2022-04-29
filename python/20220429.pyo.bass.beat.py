from pyo import *
import numpy as np

s = Server().boot()

def beatToIntervals(x):
    y = [i for i,x in enumerate(x) if x == 1] # indices of beat occurences
    z = [y[i]-y[i-1] for i in range(1,len(y))] # intervals of occurences
    z.insert(0, y[0]) # insert onset of first beat
    return z

class BassBeat:
  def __init__(self, root=49, notes=[0], beat=[1,1], time=.125, dur=0.25, attack=0.016, sustain=0.5, mul=1.0) -> None:
    self.idx = 0
    self.root = root
    self.notes = notes
    self.intervals = beatToIntervals(beat)

    self.mul = Sig(mul)
    self.mul.ctrl(title="Amplitude")

    self.seq = Seq(time=time, seq=self.intervals, poly=1, onlyonce=False, speed=1).play()

    self.env = Adsr(attack=attack, decay=attack, sustain=sustain, release=attack, dur=dur, mul=self.mul)
    self.env.ctrl()

    self.table = HarmTable([.8, 1, .8 , .5, .3, .2, .1, .05])
    self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])
    self.osc.out()

    self.tf = TrigFunc(self.seq, function=self.env.play)

#        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
beat = [ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 ]
note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]

b = BassBeat(root=25, notes=note, beat=beat, time=.1, dur=0.1, attack=0.02, sustain=0.5, mul=.3)

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
