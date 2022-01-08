from pyo import *

s = Server().boot()

t = HarmTable([1,0,.33,0,.2,0,.143,0,.111])
a = Osc(table=t, freq=[250,251], mul=.2).out()

def pat():
    f = random.randrange(200, 401, 25)
    a.freq = [f, f+1]

def scale():
    # pentatonic scales
    major = [0,2,4,7,9,12]
    minor = [0,3,5,7,10,12]
    m = random.choice(major)
    f = midiToHz(m + 48)
    a.freq = [f, f+1]

p = Pattern(scale, .125)
p.play()
s.gui(locals())