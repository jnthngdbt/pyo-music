from pyo64 import *

s = Server().boot()
s.start()

N = 1024

a = Noise(.25).mix(2)
fin = FFT(a, size=N, overlaps=4, wintype=2)

# t = ExpTable([(0,0),(3,0),(10,1),(20,0),(30,.8),(50,0),(70,.6),(150,0),(512,0)], size=512)
# amp = TableIndex(t, fin["bin"])

dv = 1
d = Sig(dv)
d.ctrl([SLMap(0., 9., 'lin', "value", 8.)], "D")

print(d.value)
print([(0,0),(10-dv,0),(10,1),(20-dv,0),(20,.8),(30-dv,0),(30,.6),(40-dv,0),(512,0)])

t = ExpTable([(0,0),(10-dv,0),(10,1),(20-dv,0),(20,.8),(30-dv,0),(30,.6),(40-dv,0),(512,0)], size=512)
amp = TableIndex(t, fin["bin"])

# t = HannTable(size=int(N/2))
# amp = TableIndex(t, fin["bin"])

re = fin["real"] * amp
im = fin["imag"] * amp
fout = IFFT(re, im, size=N, overlaps=4, wintype=2).mix(2).out()

Spectrum(fout, size=4096)

def changeSpectrum():
  if hasattr(d.value, 'value'):
    print(d.value.value)
    dd = int(d.value.value)
    ls = [(0,0),(10-dd,0),(10,1),(20-dd,0),(20,.8),(30-dd,0),(30,.6),(40-dd,0),(512,0)]
    print(t.list)
    print(ls)
    t.list = ls

clk = Pattern(changeSpectrum, 1).play()

s.gui(locals())
