from pyo import *
import numpy as np

s = Server().boot()

#         X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X
beat = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]

def beatToIntervals(x):
    y = [i for i,x in enumerate(x) if x == 1] # indices of beat occurences
    z = [y[i]-y[i-1] for i in range(1,len(y))] # intervals of occurences
    z.insert(0, y[0]) # insert onset of first beat
    return z

class KickBeat:
  def __init__(self, intervals=[0], dur=0.25, mul=1) -> None:
    self.peakFreq = 400

    self.seq = Seq(time=dur, seq=intervals, poly=1, onlyonce=False, speed=1).play()
    self.ampenv = LinTable([(0,0.0000),(600,1.0000),(6564,0.6788),(8191,0.0000)])
    self.pitchenv = LinTable([(0,0.0000),(340,1.0000),(804,0.2364),(8192,0.0667)])
    self.ampenv.graph(title="Amplitude")
    self.pitchenv.graph(title="Pitch")
    self.amp = TrigEnv(self.seq, table=self.ampenv, dur=dur, mul=mul)
    self.pitch = TrigEnv(self.seq, table=self.pitchenv, dur=dur, mul=self.peakFreq)
    self.osc = Sine(freq=self.pitch, mul=self.amp).mix(2)
    self.osc.out()

k = KickBeat(intervals=beatToIntervals(beat), dur=0.2, mul=0.2)

s.start()
s.gui(locals())
