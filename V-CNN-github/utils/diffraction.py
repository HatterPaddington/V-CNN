import numpy as np
import torch

def fresnel_diffraction(
        inpt, wavelength, L, distance
):
    inpt = torch.exp(1j * inpt)

    # wave number k
    wave_num = 2 * torch.pi / wavelength

    # the axis in frequency space
    fx = 1 / L

    x = np.linspace(-inpt.shape[0] / 2, inpt.shape[0] / 2 - 1, inpt.shape[0])
    fx = fx * x
    [Fx, Fy] = np.meshgrid(fx, fx)


    # the propagation function
    impulse_q = np.exp(
        (1j * wave_num * distance) *
        (1 - wavelength ** 2 * (Fx ** 2 + Fy ** 2)) ** 0.5
    )

    impulse_q = torch.from_numpy(impulse_q)

    part1 = torch.fft.fft2(inpt)
    part2 = torch.fft.ifftshift(impulse_q)
    diffraction = torch.fft.ifft2(part1 * part2)
    intensity = torch.abs(diffraction) * torch.abs(diffraction)

    return intensity / torch.max(intensity)
