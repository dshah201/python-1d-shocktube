import numpy as np
import matplotlib.pyplot as plt

left = [1, 1, 0]
right = [.125, .1, 0]
gamma = 1.4
n = 500
prim = np.zeros((n, 3))
x = [0] * n

b = 0
for p in x:
    if b == 0:
        x[0] = 1 / (2 * n)
    else:
        x[b] = (1 / n) + x[b - 1]
    b = b + 1

a = 0
for i in prim:
    if x[a] < 0.5:
        i[0] = left[0]
        i[1] = left[1]
        i[2] = left[2]
    else:
        i[0] = right[0]
        i[1] = right[1]
        i[2] = right[2]
    a = a + 1

# plt.plot(x, prim[:,0])
# plt.plot(x, prim[:,1])
# plt.plot(x, prim[:,2])

cons = np.zeros((n, 3))
c = 0
for num in cons:
    if x[c] < .5:
        num[0] = prim[c][0]
        num[1] = prim[c][0] * left[2]
        num[2] = prim[c][0] * (.5 * (left[2] ** 2) + left[1] / ((gamma - 1) * left[0]))
    else:
        num[0] = prim[c][0]
        num[1] = prim[c][0] * right[2]
        num[2] = prim[c][0] * (.5 * (right[2] ** 2) + right[1] / ((gamma - 1) * right[0]))
    c = c + 1

cfl = .5
dx = 1 / n
t = 0
time = 0

prim = np.vstack([prim[0], prim, prim[-1]])
cons = np.vstack([cons[0], cons, cons[-1]])

while time <= .2:
    prim[-1] = prim[-2]
    prim[0] = prim[1]
    cons[-1] = cons[-2]
    cons[0] = cons[1]

    density = cons[:, 0]
    u = cons[:, 1] / density
    pressure = ((gamma - 1) * cons[:, 2] - (gamma - 1) * .5 * density * (u ** 2))

    c = np.sqrt(gamma * pressure / density)
    t = (cfl * dx) / (np.max(np.abs(u) + c))

    densR = density[1:]
    uR = u[1:]
    densL = density[:-1]
    uL = u[:-1]
    cL = c[:-1]
    cR = c[1:]
    pressureL = pressure[:-1]
    pressureR = pressure[1:]

    max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    densflux = .5 * (densR * uR + densL * uL) - .5 * (max) * (densR - densL)
    cons[1:-1, 0] = cons[1:-1, 0] - t * (-densflux[:-1] + densflux[1:]) / dx
    momflux = .5 * ((densR * (uR ** 2) + pressureR) + (densL * (uL ** 2) + pressureL)) - .5 * (max) * (
                densR * uR - densL * uL)
    cons[1:-1, 1] = cons[1:-1, 1] - t * (-momflux[:-1] + momflux[1:]) / dx
    energyflux = .5 * ((densR * (.5 * (uR ** 2) + pressureR / ((gamma - 1) * (densR))) * uR + pressureR * uR) + (
                densL * (.5 * (uL ** 2) + pressureL / ((gamma - 1) * (densL))) * uL + pressureL * uL)) - .5 * (max) * (
                             (densR * (.5 * (uR ** 2) + pressureR / ((gamma - 1) * (densR)))) - (
                                 densL * (.5 * (uL ** 2) + pressureL / ((gamma - 1) * (densL)))))
    cons[1:-1, 2] = cons[1:-1, 2] - t * (-energyflux[:-1] + energyflux[1:]) / dx

    time = time + t

density = cons[:, 0]
u = cons[:, 1] / density
pressure = ((gamma - 1) * cons[:, 2] - (gamma - 1) * .5 * density * (u ** 2))

ypoints = density[1:-1]
plt.plot(x, ypoints)

ypointsu = u[1:-1]
plt.plot(x, ypointsu)

ypointsp = pressure[1:-1]
plt.plot(x, ypointsp)
