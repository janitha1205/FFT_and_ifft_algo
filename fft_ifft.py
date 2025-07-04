import numpy as np
import matplotlib.pyplot as plt


def signal1(t):

    tl = len(t)
    x = 1 * np.random.randn(tl)
    freq = 1.0
    x += 3 * np.sin(2 * np.pi * freq * np.asarray(t))

    freq = 4
    x += np.sin(2 * np.pi * freq * np.asarray(t))

    freq = 7
    x += 3.5 * np.sin(2 * np.pi * freq * np.asarray(t))
    return x


def calculate_a0(func, T, num_points):
    x = np.linspace(0, T, num_points)
    integral = np.trapz(func(x), x)
    return (2 / T) * integral


def calculate_an(func, T, n, num_points):
    x = np.linspace(0, T, num_points)
    integral = np.trapz(func(x) * np.cos((2 * np.pi * n * x) / T), x)
    return (2 / T) * integral


def calculate_bn(func, T, n, num_points):
    x = np.linspace(0, T, num_points)
    integral = np.trapz(func(x) * np.sin((2 * np.pi * n * x) / T), x)
    return (2 / T) * integral


def fourier_mag_phase(func, T, num_terms, x_values, N):

    mag = []
    ang = []
    freq = []
    a0 = calculate_a0(func, T, N)
    approximation = (a0 / 2) * np.ones_like(x_values)
    mag.append(a0 / 2)
    ang.append(0)
    freq.append(0)

    for n in range(1, num_terms + 1):
        an = calculate_an(func, T, n, N)
        bn = calculate_bn(func, T, n, N)
        mag.append(np.sqrt(an**2 + bn**2))
        ang.append(np.arctan2(an, bn))
        freq.append(n / T)

    return mag, ang, freq


def inv_f(mag, ang, freq, x_values, num_terms):
    mag1 = np.asarray(mag)
    approximation = np.zeros(np.size(x_values))
    for n in range(num_terms):
        ind = np.argmax(mag1)
        print(mag1[ind])
        approximation += mag[ind] * np.sin(
            (2 * np.pi * freq[ind] * np.asarray(x_values)) + ang[ind]
        )
        mag1[ind] = 0

    return x_values, approximation


def main():
    t = []
    dt = 0.01
    N = 1000
    for i in range(N):
        t.append(i * dt - dt)
    N_terms = 300
    mag, ang, freq = fourier_mag_phase(signal1, max(t), N_terms, t, N)
    N_terms_ap = 3
    x_values, approximation = inv_f(mag, ang, freq, t, N_terms_ap)

    plt.figure(figsize=(12, 10))

    plt.subplot(411)
    plt.plot(freq, mag, "r")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 10)

    plt.subplot(412)
    plt.plot(freq, ang, "b")
    plt.xlabel("Frequency(Hz)")
    plt.ylabel("Phase (rad)")
    plt.xlim(0, 10)

    plt.subplot(413)
    plt.plot(t, signal1(t), "r")
    plt.xlabel("Time(s)")
    plt.ylabel("real signal")

    plt.subplot(414)
    plt.plot(x_values, approximation, "y")
    plt.xlabel("Time(s)")
    plt.ylabel("Approximation")
    plt.show()


# Example usage:
if __name__ == "__main__":
    main()
