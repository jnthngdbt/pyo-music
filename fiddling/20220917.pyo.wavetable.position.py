from pyo import *
import numpy as np

s = Server().boot()
s.start()

pos = Sig(0)
pos.ctrl([SLMap(0, 1., 'lin', "value", pos.value)], "Position") # NOTE: the 'name' must be the name of attribute

# Create tables
nbTables = 15
tables = [SndTable("C:\\Users\\jgodbout\\Documents\\git\\sandbox\\python\\songs\\granule.mission.two.{}.wav".format(i)) for i in range(15)]

# Scan through tables

morph = NewTable(length=tables[0].getDur(), chnls=1)
selection = [tables[4], tables[10]]
m = TableMorph(pos, morph, selection)

f = s.getSamplingRate() / tables[0].size
osc = Osc(morph, freq=[f, f*1.01], mul=.2).out()

Spectrum(osc)

s.gui(locals())