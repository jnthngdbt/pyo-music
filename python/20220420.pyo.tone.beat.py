from pyo import *

s = Server().boot()

class ToneBeatSimple:
  def __init__(self, note, dur, decay=0.25, reverb=.3, mul=1.0) -> None:
    f = midiToHz(note)

    self.table = HarmTable([1, 0.3, 0.1, 0.02, 0.005])
    self.note = Osc(table=self.table, freq=[f, f*1.01])

    self.env = Adsr(attack=.01, decay=.9*dur, sustain=.0, release=.0, dur=dur)
    self.env.ctrl()

    self.reverb = Freeverb(self.note, size=[.79,.8], damp=.9, bal=reverb)
    self.reverb.ctrl()

    self.out = mul * self.reverb * self.env
    self.out.out()

    self.pat = Pattern(self.play, dur)
    self.pat.play()

  def play(self):
    self.env.play()

root = 27
dur = 0.25

tbSlow = ToneBeatSimple(note=root+24, dur=dur, decay=0.24, mul=0.1, reverb=.3)

s.start()
s.gui(locals())
