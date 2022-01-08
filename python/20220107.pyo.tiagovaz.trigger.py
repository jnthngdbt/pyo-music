from pyo import *
s = Server().boot()

# Builds an amplitude envelope in a linear table
env = LinTable([(0,0), (190,.8), (1000,.5), (4300,.1), (8191,0)], size=8192)
env.graph()

# Metronome provided by Beat
met = Beat(time=.125, taps=16, w1=90, w2=50, w3=30).play()

# Reads the amp envelope for each trigger from Beat
amp = TrigEnv(met, table=env, dur=met['dur'], mul=met['amp'])

# Generates a midi note for each trigger from Beat in a pseudo-random distribution
fr = TrigXnoiseMidi(met, dist=12, x1=1, x2=.3, scale=0, mrange=(48,85))
fr.ctrl()

# Receives the midi note from XnoiseMidi and scale it into C harmonic minor (try others!)
frsnap = Snap(fr, choice=[0,2,3,5,7,8,11], scale=1) # choice values are relative
frsnap.ctrl()

# This instrument receives a frequency from Snap and molde it inside an envelop from TrigEnv
lfo = Sine(freq=.05, mul=.05, add=.08)
gen = SineLoop(freq=frsnap, feedback=lfo, mul=amp*.5).out(0)

# Output the same signal with some delay in the right speaker (try a 'real' counterpoint!)
rev = Delay(gen, delay=[.25, .5], feedback=.3, mul=.8).out(1)

s.gui(locals())