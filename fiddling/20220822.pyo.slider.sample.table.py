from pyo import *
import numpy as np

s = Server().boot()
s.start()

def expand(notes=[48, 52, 55], octaves=[0,1,2]):
  x = []
  notes = np.array(notes)
  for i in octaves:
    x = np.concatenate([x, i * 12 + notes], axis=0)
  return x.tolist()
  
t = SndTable("C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\slider.spectrum.sample.wav")

root = 56 # c#
amp = [1] # [1, .6, .2]
notes = [0] # [0, 7, 12]
octaves = [0]

padFreq = 220 # A3: 220Hz, A4: 440Hz
padSize = t.size
padRatio = s.getSamplingRate() / padSize

freq = expand(octaves=octaves, notes=[x + root for x in notes])
freq = padRatio * np.array(midiToHz(freq)) / padFreq

p = []
p.append(Osc(t, freq=freq.tolist(), mul=amp).mix(2))
# p.append(Delay(p[-1], delay=[.3, .32], feedback=0.6))
# p.append(WGVerb(p[-1])), p[-1].ctrl()
# p.append(EQ(p[-1], [100,400,1600], q=5, boost=[0,0,0])), p[-1].ctrl()
p.append(MoogLP(p[-1], 20000, .5)), p[-1].ctrl()
# p.append(Pan(p[-1], 2, pan=.5, spread=1.))
p[-1].out()

Spectrum(p[-1])

s.gui(locals())