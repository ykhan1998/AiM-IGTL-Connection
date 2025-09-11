import random

import numpy as np
from scipy.optimize import curve_fit


def compute_prfs(prfs_info, prfs_roi, dicom_header_info, imgs, baseline=None, debug=False):
    # Load Manual and Header Defined Parameters
    # Manual Defined
    rhrcctrlfactor = prfs_info['rhrcctrlfactor']
    alpha = prfs_info['alpha']
    gamma = prfs_info['gamma']
    number_of_coils = prfs_info['number_of_coils']
    numberslices = prfs_info['numberslices']

    # Header Defined
    y_res = dicom_header_info['Height']
    x_res = dicom_header_info['Width']
    acquisition_time = dicom_header_info['AcquisitionTime']
    magnetic_field_strength = dicom_header_info['MagneticFieldStrength']
    echo_time = dicom_header_info['EchoTime'] / 1000

    Denominator = -alpha * gamma * magnetic_field_strength * echo_time

    if debug:
        print("alpha: " + str(alpha))
        print("gamma: " + str(gamma))
        print("fieldstrength: " + str(magnetic_field_strength))
        print("TE: " + str(echo_time))
        print("Denominator: " + str(Denominator))

    [w, h] = np.shape(imgs[0])

    if len(imgs) == 9:
        real_imgs = np.array([imgs[1], imgs[4]])
        imaginary_imgs = np.array([imgs[2], imgs[5]])
    elif len(imgs) == 2:
        real_imgs = np.array([imgs[0]])
        imaginary_imgs = np.array([imgs[1]])
    else:
        print("Wrong img number:" + str(len(imgs)))
        return None

    if debug:
        if len(imgs) == 9:
            mag_imgs = np.array([imgs[0], imgs[3]])
        elif len(imgs) == 2:
            mag_imgs = np.array([imgs[0]])
        print("real and imaginary shapes")
        print(np.shape(real_imgs))
        print(np.shape(imaginary_imgs))
        x = y = 128
        i = random.randint(0, np.shape(real_imgs)[0] - 1)
        print(mag_imgs[i, x, y])
        print(real_imgs[i, x, y])
        print(imaginary_imgs[i, x, y])
        mag = np.sqrt(real_imgs[i, x, y] ** 2 + imaginary_imgs[i, x, y] ** 2)
        print(mag)

    imgs = np.array(imgs)

    # Form Complex and Phase Images
    complex_img = real_imgs + 1j * imaginary_imgs
    phase_img = np.angle(complex_img)

    #  Check if an initial baseline already exists and if not set it
    if baseline is None:
        baseline = complex_img
        uncorrected_del_t_img = np.zeros((w, h))
        del_t_img = np.zeros((w, h))
        return del_t_img, uncorrected_del_t_img, baseline

    # Perform phase subtraction from baseline
    del_ = np.zeros_like(complex_img)
    for i in range(len(phase_img)):
        del_[i] = complex_img[i] * np.conj(baseline[i])

    # Sum multi channel data
    Rx = np.angle(del_)
    Rx = np.sum(Rx, axis=0)

    del_T_img = Rx / Denominator

    # return del_T_img, None, baseline

    # Perform B0 Phase Compensation
    uncorrected_del_t_img = del_T_img

    phase_correction = CompensateB0PhaseDrift(del_T_img, prfs_roi)
    del_T_img -= phase_correction

    if debug:
        np.save("phase_img.npy", phase_img)
        # np.save("del_phase.npy", del_phase)
        np.save("Rx.npy", Rx)
        np.save("uncorrected.npy", uncorrected_del_t_img)
        np.save("del_T_img.npy", del_T_img)

    return del_T_img, uncorrected_del_t_img, baseline


def func1(X, A, B, C):
    x = X[0]
    y = X[1]
    return (A * x) + (B * y) + C


def func2(X, A, B, C, D, E, F):
    x = X[0]
    y = X[1]
    return (A * x ** 2) + (B * y ** 2) + (C * x * y) + \
           (D * x) + (E * y) + F


def func3(X, A, B, C, D, E, F, G, H, I, J):
    x = X[0]
    y = X[1]
    return (A * x ** 3) + (B * y ** 3) + (C * x ** 2 * y) + (D * x * y ** 2) + \
           (E * x ** 2) + (F * y ** 2) + (G * x * y) + \
           (H * x) + (I * y) + J


def CompensateB0PhaseDrift(Tmap, prfs_roi):
    [w, h] = np.shape(Tmap)
    # 1. Get values within ROI
    roiValues = Tmap * prfs_roi
    [X, Y] = np.nonzero(roiValues)
    Z = roiValues[X, Y]
    # 2. Perform Polynomial Fit
    parameters, _ = curve_fit(func2, [X, Y], Z)

    # 3. Create new Matrix with fit
    model_x_data = np.linspace(0, w, w)
    model_y_data = np.linspace(0, h, h)
    x, y = np.meshgrid(model_x_data, model_y_data)

    prfsCorWPI = func2(np.array([x, y]), *parameters)
    return prfsCorWPI
