from __future__ import print_function
content='''
[SimVars.0]
Latitude=N21° 20' 47.36"
Longitude=W157° 27' 23.20"
'''

latKey = "Latitude="
longKey = "Longitude="

latstart = content.index(latKey) + len(latKey)
latend = content.find('"', latstart) + 1
longstart = content.find(longKey, latend) + len(longKey)
longend = content.find('"', longstart) + 1

lat = content[latstart:latend]
long = content[longstart:longend]

print()
print('lat ', lat)
print('long ', long)

deg, mnt, sec = [float(x[:-1]) for x in lat[1:].split()]
latVal = deg + mnt / 60 + sec / 3600

deg, mnt, sec = [float(x[:-1]) for x in long[1:].split()]
longVal = deg + mnt / 60 + sec / 3600

print()
print('latVal ', latVal)
print('longVal ', longVal)