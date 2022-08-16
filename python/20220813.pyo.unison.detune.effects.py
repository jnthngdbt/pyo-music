from pyo import *

s = Server().boot()
s.start()

f = 200
d = Sig(1) # detune
d.ctrl([SLMap(0., 10., 'lin', "value", d.value)], "Detune") # NOTE: the 'name' must be the name of attribute

nbVoicesEachSide = 3

def getUnisonFreqs(freq):
  return [freq+i*d for i in range(-nbVoicesEachSide, nbVoicesEachSide)]

t = SawTable()

md = hzToMidi(f)

x = [
  Osc(t, freq=getUnisonFreqs(midiToHz(md)), mul=.02, phase=0).out(),
  Osc(t, freq=getUnisonFreqs(midiToHz(md+7)), mul=.015, phase=0).out(),
  Osc(t, freq=getUnisonFreqs(midiToHz(md+12)), mul=.01, phase=0).out(),
  Osc(t, freq=getUnisonFreqs(midiToHz(md+16)), mul=.008, phase=0).out(),
]


s.gui(locals())
