
from pyo import *

s = Server().boot()
s.start()

# Stay below midi(30) or 60Hz for some growl.
class BigBass:
  def __init__(self, beat, f=midiToHz(25), mul=1) -> None:
    # self.env = Adsr(attack=.01, decay=0.01, sustain=0.8, release=4, dur=5)
    self.env = TrigExpseg(beat, [(0,0), (.05,1), (.1, .8), (4, 0)])

    self.p = []
    self.p.append(HarmTable([.5, 1, .7, .2])) # sub 0, main, fifth, octave
    self.p.append(Osc(self.p[-1], f))
    self.p[-1].ctrl()
    self.p.append(Chorus(self.p[-1], [1, 1.2], feedback=0)) # adds some movement
    self.p[-1].mul = mul * self.env
    self.p[-1].ctrl()
    self.p[-1].out()



trig = Beat(time=.125, w1=80, w2=0, w3=0).play()
B = BigBass(trig, midiToHz(25), mul=.2) # better if around midi(25)
B = BigBass(trig, midiToHz(37), mul=.1) # better if around midi(37)

Spectrum(B.p[-1], size=8192)

s.gui(locals())