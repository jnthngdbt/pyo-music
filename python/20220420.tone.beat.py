from pyo import *

s = Server().boot()

class ToneBeatSimple:
  def __init__(self, note=48, bps=3.0, mul=1.0) -> None:
    self.table = HarmTable([1, 0.3, 0.1, 0.02, 0.005])
    f = midiToHz(note)
    self.note = Osc(table=self.table, freq=[f, f*1.01])

    self.env = Adsr(attack=.012, decay=.23, sustain=.2, release=.1, dur=.2)
    self.env.ctrl()

    self.out = mul * self.note * self.env
    self.out.out()

    self.pat = Pattern(self.play, 1./bps)
    self.pat.play()

  def play(self):
    self.env.play()

class Peak:
  def __init__(self, note=48, mul=1.0) -> None:
    self.noise1 = PinkNoise()
    self.noise2 = PinkNoise()
    self.band = Biquadx([self.noise1, self.noise2], freq=midiToHz(note), q=12, type=2, stages=2, mul=mul).out()
    self.band.ctrl()

pk = Peak(note=85, mul=0.1)
tb = ToneBeatSimple(note=49, bps=4, mul=0.1)

s.start()
s.gui(locals())