from pyo import *
import numpy as np
from pyoband import band

s = Server().boot()

class DistoGuit (band.Track):
    def __init__(self, bendFreq=0, drive=.5, adsr=[.01,.1,.8,.1], tone=[.6, .8, 1., .5, .3, .2, .1], reverb=0., **kwargs):
        super().__init__(**kwargs)

        self.env = Adsr(attack=adsr[0], decay=adsr[1], sustain=adsr[2], release=adsr[3], dur=self.dur, mul=self.mul)
        # self.env.ctrl()

        self.pitchFactor = Sine(freq=bendFreq).range(1, 1.01)

        self.table = HarmTable(tone)
        self.osc = Osc(table=self.table, freq=midiToHz(25), mul=self.env, phase=[0,.4])

        self.disto = Disto(self.osc, drive=drive, slope=.8, mul=self.mul)

        self.effect = Freeverb(self.disto, size=0.8, damp=0.5, bal=reverb)
        self.effect.out()

    def play(self, note=0):
        self.osc.freq = self.pitchFactor * midiToHz(note)
        self.env.play()

time = 4
root = 36
reverb = .6

M = band.Maestro(time=time, tracks=[
        # DistoGuit(name="Power Chord", mul = .8, root = root, dur=time, reverb=reverb, drive=.5,#tone=[1], #, .2, .05, .01],
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 1 ],
        #     note = [ 0 ]
        # ),
        DistoGuit(name="Pinch Harmonic", mul = .12, root = root+48, dur=time, reverb=reverb, bendFreq=4, drive=.98, tone=[1], #, .2, .05, .01],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ],
            note = [ 0 ]
        ),
])

s.start()
s.gui(locals())
