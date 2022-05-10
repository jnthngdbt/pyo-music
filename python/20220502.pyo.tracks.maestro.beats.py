from pyo import *
import numpy as np
from pyoband import band

s = Server().boot()

root = 28
dur = .15

M = band.Maestro(time=dur, nbSectionsToggle=2, tracks=[
        band.BassBeat(name="Bass Beat", mul = 0.08, root = root, adsr=[.02, .02, .8, .04], tone=[.4, 1, .6, .2, .1],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        ),
        band.BassBeat(name="High Beep", mul = 0.012, root = root+36, adsr=[.02, .02, .5, .04], tone=[1],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0 ],
            note = [ 12 ]
        ),
        band.BassBeat(name="Low Mid Beat", mul = 0.03, root = root+12, adsr=[.02, .02, .5, .04], tone=[1, .8, .1 ],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0,17, 0, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0,16, 0, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0, 0,14, 0 ]
        ),
        band.BassBeat(name="Mid Beat", mul = 0.018, root = root+36, adsr=[.02, .02, .5, .04], tone=[1 ],
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 ],
            note = [ 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0, 0,12, 0, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 9, 0, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0,11, 0, 0,11, 0, 0,11, 0, 0,11, 0, 0,11, 0 ]
        ),
        band.Kick(name="Kick", mul = 0.2, dur = dur,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0 ]
        ),
        band.Snare(name="Snare", mul = 0.3, dur = dur, cutoff = 1000,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 0, 0, 0, 0, 1, 0, 0, 0 ]
        ),
        # band.Arpeggio(name="Arpeggio 1", mul=0.003, root=root+48, notes=band.expand([0,4,7], octaves=[0,1]), dur=dur, sustain=.75, doMirror=True,
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 1 ]
        # ),
        band.Arpeggio(name="Arpeggio 2", mul=0.006, root=root+36, notes=band.expand([0,4,5,7], octaves=[0,1,2]), dur=dur, sustain=.75, doMirror=True,
            #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
            beat = [ 1 ]
        ),
        # band.Arpeggio(name="Arpeggio 3", mul=0.006, root=root+36, notes=band.expand([0,4,7,0], octaves=[0,1,2,3]), dur=dur, sustain=.75, doMirror=False,
        #     #        |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x  |  x  x  x  X  x  x  x  X  x  x  x  X  x  x  x
        #     beat = [ 1 ]
        # ),
])

s.start()
s.gui(locals())
