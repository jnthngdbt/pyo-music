from pyo import *

# s = Server()
# s.deactivateMidi()
# s.boot()
# def midicall(status, data1, data2):
#     print(status, data1, data2)
# listen = MidiListener(midicall, 1)
# listen.start()
# s.gui(locals()) # Prevents immediate script termination.

# Set Up Server
s = Server()
s.allowMicrosoftMidiDevices()
s.setMidiInputDevice(1) # Change as required
s.boot()
s.start()

# Set Up MIDI
midi = Notein()

# ADSR
amp = MidiAdsr(midi['velocity'])

# Pitch
pitch = MToF(midi['pitch'])

# Table
wave = SquareTable()

# Osc
osc = Osc(wave, freq=pitch, mul=amp)

# FX
verb = Freeverb(osc).out()

### Go
osc.out()
s.gui(locals()) # Prevents immediate script termination.