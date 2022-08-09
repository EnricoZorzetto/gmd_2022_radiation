import numpy as np
import matplotlib.pyplot as plt

APP = np.array( [0.1, 0.2, 0.3, 0.5] )

# FP = np.array( [0.8, 1.0, 1.2, 1.5] )


FP = np.linspace(0.9, 1.6)

colors = ['-r', '-g', '-b', '-c', 'y']


plt.figure()
for ia, ap in enumerate(APP):
    A3D = 1.0 - FP * ( 1 - ap)
    # plt.plot(FP, A3D - ap, colors[ia], label=r'$\alpha_{{pp}}$ = {}'.format(ap))
    plt.plot(FP, A3D, colors[ia], label=r'$\alpha_{{pp}}$ = {}'.format(ap))
    plt.plot(FP, np.zeros(np.size(A3D)), '--k')
    plt.xlabel(r'$f_{p}$')
    # plt.ylabel(r'$\alpha_{3d} - \alpha_{pp}$')
    plt.ylabel(r'$\alpha_{3d}$')
    plt.legend()
    # plt.plot(FP)

plt.show()


plt.figure()
for iff, fp in enumerate(FP):
    A3D = 1.0 - fp * ( 1 - APP)
    plt.plot(APP, A3D, colors[iff], label=r'$f{{p}}$ = {}'.format(fp))
    plt.plot(APP, APP, '--k')
    plt.legend()
    plt.xlabel(r'$\alpha_{pp}$')
    plt.ylabel(r'$\alpha_{3d}$')
    # plt.plot(FP)

plt.show()
