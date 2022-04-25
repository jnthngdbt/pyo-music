from pyo import *
import numpy as np

s = Server().boot()

#         X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X
defseq = [1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1]
defint = [0,8,2,3,3]

class KickBeat:
  def __init__(self, intervals=defint, time=0.25) -> None:
    self.seq = Seq(time=time, seq=intervals, poly=1, onlyonce=False, speed=1).play()
    self.env = CosTable([(0,0),(300,1),(1000,.3),(8191,0)])
    self.env.graph()
    self.amp = TrigEnv(self.seq, table=self.env, dur=.25, mul=1)
    self.noiseL = PinkNoise()
    self.noiseR = PinkNoise()
    self.lp = ButBP([self.noiseL, self.noiseR], freq=200, mul=self.amp).out()
    self.lp.ctrl()

k = KickBeat(intervals=defint, time=0.2)

s.start()
s.gui(locals())