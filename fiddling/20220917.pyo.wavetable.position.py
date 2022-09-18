from pyo import *
import numpy as np

s = Server().boot()
s.start()

class Pad:
  def __init__(self):
    self.size = 262144
    self.basefreq = 440
    self.ratio = s.getSamplingRate() / self.size
    self.table = PadSynthTable(basefreq=self.basefreq, size=self.size, 
      spread=1, # for slight dissonnance
      bw=50, # def: 50, pure > noisy
      bwscl=1., # def: 1, breathing
      damp=.8) # def: 0.7, mellow/bright

  def freq(self, midi=48):
    f = midiToHz(midi)
    f = np.array(f)
    f = self.ratio * f / self.basefreq
    return f.tolist()

pos = Sig(0)
pos.ctrl([SLMap(0, 1., 'lin', "value", pos.value)], "Position") # NOTE: the 'name' must be the name of attribute

fmin = midiToHz(37)
fmax = midiToHz(85)
freq = fmin + (fmax-fmin) * pos

# Create tables
nbTables = 4
size = 262144
basefreq = 440
ratio = s.getSamplingRate() / size
tables = [PadSynthTable(basefreq=basefreq, size=size, damp=float(.9-.6*x), bw=float(20 + 50*x)) for x in np.arange(0,1,1./32.)]

# Scan through tables
morph = NewTable(length=size/s.getSamplingRate(), chnls=1)
m = TableMorph(pos, morph, tables)

f = ratio * freq / basefreq
osc = Osc(morph, freq=[f,f*1.01], mul=.2).out()

s.gui(locals())