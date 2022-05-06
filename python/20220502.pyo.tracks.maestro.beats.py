from pyo import *
import numpy as np
from pyoband import band

s = Server().boot()

root = 25
dur = .2

M = band.Maestro(time=0.125, tracks=[
        band.BassBeat(name="Bass Beat", mul = 0.2, root = root, attack=0.02, sustain=0.5, tone=[.8, 1, .8, .5, .2],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        band.BassBeat(name="High Beep", mul = 0.01, root = root+36, tone=[1],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0 ],
            note = [ 12 ]
        ),
        band.BassBeat(name="Low Mid Beat", mul = 0.038, root = root+12, tone=[1, .8 , .3 ],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0 ]
        ),
        band.BassBeat(name="Mid Beat", mul = 0.02, root = root+36, tone=[1 ],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,11, 0, 0,11, 0, 0,11, 0, 0,11, 0, 0,11, 0 ]
        ),
        band.Kick(name="Kick", mul = 0.4, dur = dur,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 ]
        ),
        band.Snare(name="Snare", mul = 0.4, dur = dur, cutoff = 1000,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 0, 0, 1, 0, 0, 0 ]
        ),
        band.Arpeggio(name="Arpeggio", mul=0.008, root=root+48, notes=band.expand([0,4,7,11], octaves=[0,1,2,3]), dur=dur, sustain=.75, doMirror=False,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ]
        ),
])

s.start()
s.gui(locals())
