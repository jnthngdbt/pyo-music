from pyo import *

s = Server().boot()

t = HarmTable([1, 0, .1, 0, .02])
# t = HarmTable([1,0,.33,0,.2,0,.143,0,.111])
f = midiToHz(49)
a = Osc(table=t, freq=[f, f+1])

dt = 0.003
env = Adsr(attack=.01, decay=.2, sustain=.1, release=.1, dur=.2)

x = 0.1 * a * env
d = Delay(x, delay=[.00, .02], feedback=0)

x.out()

def note():
    env.play()

p = Pattern(note, 0.25)
p.play()

class Peak:
  def __init__(self, note=48, mul=1.0) -> None:
    self.noise1 = PinkNoise()
    self.noise2 = PinkNoise()
    self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=12, type=2, stages=2, mul=mul).out()

pk = Peak(note=85, mul=0.1)

s.start()
s.gui(locals())