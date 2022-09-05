
from pyo import *

s = Server().boot()
s.start()

# Adsr can start at any point there wont be clicks
# Defined with durations, so first point is not necessarily 0
class EnvAdsr:
  def __init__(self, f, mul=1) -> None:
    self.env = Adsr(attack=3, decay=0., sustain=1., release=3, dur=0)
    self.env.setExp(2)

    self.p = []
    self.p.append(HarmTable([.5, 1, .7, .2])) # sub 0, main, fifth, octave
    self.p.append(Osc(self.p[-1], f))
    self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0, bal=0.)) # adds some movement
    self.p[-1].mul = mul * self.env
    self.p[-1].ctrl()
    self.p[-1].out()

    self.toggle = True

  def play(self):
    self.env.play() if self.toggle else self.env.stop()
    self.toggle = not self.toggle

# Can click if starting envelope before back to 0
class EnvSegs:
  def __init__(self, beat, f, mul=1) -> None:
    self.env = TrigLinseg(beat, [(0,0), (.1,1), (.2, .8), (4, 0)])

    self.p = []
    self.p.append(HarmTable([.5, 1, .7, .2])) # sub 0, main, fifth, octave
    self.p.append(Osc(self.p[-1], f))
    self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0, bal=0.)) # adds some movement
    self.p[-1].mul = mul * self.env
    self.p[-1].out()


# trig = Beat(time=.25, w1=100, w2=0, w3=0).play()
# B = EnvSegs(trig, midiToHz(37), mul=.2)

B = EnvAdsr(midiToHz(37), mul=.1)
p = Pattern(B.play, time=2).play()

Spectrum(B.p[-1], size=8192)

s.gui(locals())