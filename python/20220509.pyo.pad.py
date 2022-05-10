from pyo import *
import numpy as np
from pyoband import band

s = Server().boot()

class Pad (band.Track):
    def __init__(self, reverb=0., **kwargs):
        super().__init__(**kwargs)

        # For high note (85-97)
        # - for soft high with bit of air: bw=40, damp=0.2, bwscl=2.0

        # For low note (36)
        # - for low string: damp=0.9 (grittier)
        # - for low string: damp=0.85, bw = 40, bwscl=1.1
        # - for low bass: damp=0.5
        # - for low growl: midi 25, damp=0.9 (0.95 more open)
        self.size = 262144
        self.basefreq = 440
        self.ratio = s.getSamplingRate() / self.size
        self.table = PadSynthTable(basefreq=self.basefreq, size=self.size)

        self.trig = Trig()
        self.env = LinTable([(0,0.0000),(1,0.8545),(1949,0.9273),(8191,0.5939),(8192,0.0000)])
        self.amp = TrigEnv(self.trig, table=self.env, dur=self.dur, mul=self.mul)

        self.env.graph()

        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.amp, phase=[0,.4])
        self.effect = Freeverb(self.osc, size=0.5, damp=0.5, bal=reverb)
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = self.freq(note)
        self.trig.play()

    def freq(self, midi=48):
        f = midiToHz(midi)
        f = np.array(f)
        f = self.ratio * f / self.basefreq
        return f.tolist()

class PadLowGrowl (Pad): # for around midi 25
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table.setBw(30, generate=False) # 20: org, 50: def strings, 70: dreamy more highs
        self.table.setBwScl(1, generate=False) # 1: def string, 2: dreamy, 3: flute/wind
        self.table.setDamp(0.9, generate=False) # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
        self.table.setNharms(64, generate=True)

class PadLowString (Pad):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table.setBw(50, generate=False) # 20: org, 50: def strings, 70: dreamy more highs
        self.table.setBwScl(1, generate=False) # 1: def string, 2: dreamy, 3: flute/wind
        self.table.setDamp(0.9, generate=False) # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
        self.table.setNharms(64, generate=True)

class PadHighStringSoft (Pad):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table.setBw(60, generate=False) # 20: org, 50: def strings, 70: dreamy more highs
        self.table.setBwScl(1, generate=False) # 1: def string, 2: dreamy, 3: flute/wind
        self.table.setDamp(0.2, generate=False) # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
        self.table.setNharms(64, generate=True)

class PadHighString (Pad):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table.setBw(40, generate=False) # 20: org, 50: def strings, 70: dreamy more highs
        self.table.setBwScl(1, generate=False) # 1: def string, 2: dreamy, 3: flute/wind
        self.table.setDamp(0.5, generate=False) # 0.7: def, 1: big high synth, 0.5: mellow // amp = pow(damp, n)
        self.table.setNharms(64, generate=True)

time = 4
root = 25
reverb = .2

M = band.Maestro(time=time, tracks=[
        PadLowGrowl(name="Low Growl", mul = 0.9, root = root, dur=time, reverb=reverb,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 9, 5, 0, 7 ]
        ),
        PadLowString(name="Low Mid", mul = 0.4, root = root+12, dur=time, reverb=reverb,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 4, 5, 4, 2, 4, 5, 7, 11 ]
        ),
        PadHighString(name="High Mid", mul = 0.08, root = root+60, dur=time, reverb=reverb,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0 ]
        ),
        PadHighStringSoft(name="High", mul = 0.02, root = root+72, dur=time, reverb=reverb,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 7 ]
        ),
])

s.start()
s.gui(locals())
