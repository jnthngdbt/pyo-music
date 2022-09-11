from pyo import *

s = Server().boot()

# A four-streams oscillator to produce a chord.
osc = SineLoop(freq=[0, 0, 0], feedback=0.05, mul=[0.4, 0.1, 0.05])
rev = WGVerb(osc.mix(2), feedback=0.8, cutoff=4000, bal=0.2).out()

def set_osc_freqs(notes):
    # PyoObject.set() method allow to change the value of an attribute
    # with an audio ramp to smooth out the change.
    osc.set(attr="freq", value=midiToHz(notes), port=0.005)


# The sequence of functions (some call set_osc_freqs to change the notes).
def event_0():
    set_osc_freqs([46, 53, 58])


def event_1():
    pass


def event_2():
    set_osc_freqs([42, 49, 54])


def event_3():
    pass


def event_4():
    set_osc_freqs([49, 56, 61])


def event_5():
    pass


def event_6():
    set_osc_freqs([44, 51, 56])


def event_7():
    pass


# Integer generator (more about triggers in section 12-triggers)
metro = Metro(time=0.5).play()
count = Counter(metro, min=0, max=8)

# Score calls the function named "event_" + count. (if count is 3,
# function named "event_3" is called without argument.
score = Score(count, fname="event_")

s.gui(locals())