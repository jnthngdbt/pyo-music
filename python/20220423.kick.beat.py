from pyo import *
import numpy as np

s = Server().boot()

#         X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X   x   x   x   X
defseq = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
defint = [0, 8, 2, 3, 3]


class KickBeat:
  def __init__(self, intervals=defint, time=0.25) -> None:
    self.seq = Seq(time=time, seq=intervals, poly=1, onlyonce=False, speed=1).play()
    self.env = CosTable([(0,0.0000),(339,0.9636),(1591,0.0000),(8191,0.0000)])
    self.env.graph()
    self.amp = TrigEnv(self.seq, table=self.env, dur=.25, mul=10)
    self.noiseL = BrownNoise()
    self.noiseR = BrownNoise()
    self.lp = ButBP([self.noiseL, self.noiseR], freq=40, q=4, mul=self.amp)
    self.lp.ctrl()
    self.effect = Freeverb(self.lp, size=0.5, damp=0.5, bal=0.5, mul=1)
    self.effect.ctrl()
    # self.effect.ctrl([
    #     SLMap(0, 10, 'lin', 'mul', 10)
    # ])
    self.effect.out()


k = KickBeat(intervals=defint, time=0.2)

s.start()
s.gui(locals())
