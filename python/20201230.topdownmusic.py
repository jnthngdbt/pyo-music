'''

# Chords

## Triads: root - third - fifth

In the key of C:
Major: C-E-G
Minor: C-Eb-G

Note: what determines minor/major is the third; a 3-notes power chord is neither major nor minor.

# Progressions

vi-IV-I-V

In the key of C:
I   - C major
ii  - D minor
iii - E minor
IV  - F major
V   - G major
vi  - A minor
vii - B minor (vii0 for diminished triad)

# Inversions

root chord: C-E-G
1st inversion: E-G-C
2nd inversion: G-C-E (*)

All together: C-E-G-C-E

(*) 2nd inversion is equivalent to a 4 notes power chord without the first (bass) note

'''

class Chord:
     def __init__(self, amp):
         if not isinstance(amp, list): amp = [amp]
         self.amp = amp

powr = [1,0,1,1,0]
root = [1,1,1,0,0]
inv1 = [0,1,1,1,0]
inv2 = [0,0,1,1,1]
note = [1,0,0,0,0]
octv = [1,0,1,0,0]

# class I (Chord):
# class ii (Chord):
# ii = Chord()
# iii = Chord()
# IV = Chord()
# V = Chord()
# vi = Chord()
# vii = Chord()

# I (powr) + gjgjjggjgjgjgjgjjggj gjhk kgjh ghj jgh ghj gjh jhg jgh

