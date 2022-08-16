from pyo import *
import numpy as np

s = Server().boot()
s.start()

f = 100
d = Sig(.35) # detune
d.ctrl([SLMap(0., 10., 'lin', "value", d.value)], "Detune") # NOTE: the 'name' must be the name of attribute

nbVoicesEachSide = 8
nbVoices = nbVoicesEachSide * 2

def getUnisonFreqs(freq):
  return [freq+i*d for i in range(-nbVoicesEachSide, nbVoicesEachSide)]

t = SawTable()

md = hzToMidi(f)

#
# AH, to eliminate interference beating, each voice must have a random phase.
#
x = [
  Osc(t, freq=getUnisonFreqs(midiToHz(md)), mul=.02, phase=np.random.random(nbVoices).tolist()),
  Osc(t, freq=getUnisonFreqs(midiToHz(md+7)), mul=.015, phase=np.random.random(nbVoices).tolist()),
  Osc(t, freq=getUnisonFreqs(midiToHz(md+12)), mul=.01, phase=np.random.random(nbVoices).tolist()),
  # Osc(t, freq=getUnisonFreqs(midiToHz(md+16)), mul=.008, phase=np.random.random(nbVoices).tolist()).out(),
]

p = []
p.append(Mix(x, 2))
# pipeline.append(Freeverb(pipeline[-1], size=[.99,.95], damp=.5, bal=0.5)), pipeline[-1].ctrl()
p.append(Delay(p[-1], delay=[.15, .2], feedback=.0)), p[-1].ctrl()
p.append(MoogLP(p[-1], freq=20000, res=0.0)), p[-1].ctrl()
p[-1].out()

s.gui(locals())
