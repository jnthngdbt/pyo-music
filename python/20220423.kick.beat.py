from pyo import *
import numpy as np

s = Server().boot()

#         X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X
defseq = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
defint = [0, 8, 2, 3, 3]


class KickBeat:
  def __init__(self, intervals=defint, dur=0.25, mul=1) -> None:
    self.peakFreq = 400

    self.seq = Seq(time=dur, seq=intervals, poly=1, onlyonce=False, speed=1).play()
    self.ampenv = LinTable([(0,0),(600,1),(8191,0)])
    self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(5043,0.0848),(8192,0.0667)])
    self.ampenv.graph(title="Amplitude")
    self.pitchenv.graph(title="Pitch")
    self.amp = TrigEnv(self.seq, table=self.ampenv, dur=dur, mul=mul)
    self.pitch = TrigEnv(self.seq, table=self.pitchenv, dur=dur, mul=self.peakFreq)
    self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
    self.osc.out()


k = KickBeat(intervals=defint, dur=0.2, mul=0.15)

s.start()
s.gui(locals())
