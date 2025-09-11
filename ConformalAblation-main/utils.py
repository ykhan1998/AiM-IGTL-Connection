import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydicom import dcmread

from constants import STATUS_1, STATUS_2, STATUS_3, STATUS_4, STATUS_VALUES


def compute_status_map(ablated_map, lesion_mask):
    h, w = np.shape(ablated_map)
    status_map = np.zeros((h, w), dtype='uint8')
    status_value = 0

    status_lookup = {
        (0, 0): (STATUS_1, STATUS_VALUES["healthy_unablated"]),
        (0, 1): (STATUS_2, STATUS_VALUES["healthy_ablated"]),
        (1, 0): (STATUS_3, STATUS_VALUES["lesion_unablated"]),
        (1, 1): (STATUS_4, STATUS_VALUES["lesion_ablated"])
    }

    # Vectorized operations
    lesion_ablated_combo = np.stack((lesion_mask, ablated_map), axis=-1)
    for (lesion, ablated), (status, value) in status_lookup.items():
        mask = np.all(lesion_ablated_combo == [lesion, ablated], axis=-1)
        status_map[mask] = status
        status_value += np.sum(mask) * value

    return status_map, status_value
            
    # DiffAblationAITemplate.m L92
    # Katie's state definition: int8(step_ob) - this.downsampled_desired_lesion_pattern
    # 1(cem43>thresh) - 0(healthy) = 1
    # 1(cem43>thresh) - 1(tumor) = 0
    # 0(cem43<thresh) - 0(healthy) = 0
    # 0(cem43<thresh) - 1(tumor) = -1
    # Information LOSS!        


def comsol_acpr_parser(acpr_folder_path):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(r'matlab_functions'), nargout=0)

    for acpr_filename in glob.glob(acpr_folder_path + "/*.csv"):
        print("Parsing " + acpr_filename)
        eng.ImportAcousticPressureField(acpr_filename, nargout=0)
        print("Done.")


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def center_img(img):
    hh, ww = img.shape
    
    # Get contours (presumably just one around the nonzero pixels) 
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    if not contours:
        return None  # Return None if no contours found
    
    # Find bounding rectangle of all contours
    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    # Recenter
    startx = max(0, (ww - w) // 2)
    starty = max(0, (hh - h) // 2)
    endx = min(ww, startx + w)
    endy = min(hh, starty + h)
    
    result = np.zeros_like(img)
    result[starty:endy, startx:endx] = img[y:y+h, x:x+w]

    return result


def dicom_editor(filename, flip=True):
    dicom_file = dcmread(filename, specific_tags=None)
    img = dicom_file.pixel_array
    print(f"Original size: {img.shape}")

    if flip:
        flipped_img = cv2.flip(img, 1)
        img = cv2.flip(flipped_img, 0)

    img[img == 2] = 0 
    img[img == 4] = 1

    # # 1mm -> 0.7mm, pixel number *= 1.43 (1mm / 0.7mm)
    # h, w = int(img.shape[0] * 1.43), int(img.shape[1] * 1.43)
    # print(f"Resized: {(h, w)}")
    # resized_img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    # plt.imshow(resized_img)
    # plt.show()

    # if h > 144:
    #     start_h = int((h - 144)/2)
    #     resized_img = resized_img[start_h:start_h + 144, :]
    # if w > 144:
    #     start_w = int((w - 144)/2)
    #     resized_img = resized_img[:, start_w:start_w + 144]
    # print(f"Cropped: {resized_img.shape}") 

    padded_img = padding(img, 100, 100)
    print(f"Padded: {padded_img.shape}") 

    u8_img = padded_img.astype(np.uint8)
    centered_img = center_img(u8_img)
    print(f"Centered: {centered_img.shape}") 

    centered_img[centered_img > 0] = 1
    centered_img[centered_img <= 0] = 0

    np.save(filename.replace("dcm", "npy"), centered_img)

    # Save jpeg for visualization.
    centered_img[centered_img == 1] = 255
    im = Image.fromarray(centered_img)
    im.save(filename.replace("dcm", "jpeg"))

    # Save DICOM.
    # dicom_file.PixelData = img.tobytes()
    # dicom_file.save_as(filename.split(".")[0] + "IMG0001_edited")


def compute_radial_diff(status_map):
    height, width = status_map.shape
    center = (width // 2, height // 2)
    
    # Reconstuct lesion mask and ablation pattern from status map.
    lesion_mask = ((status_map == STATUS_4) | (status_map == STATUS_3)).astype(np.uint8) * 255
    ablation_pattern = ((status_map == STATUS_2) | (status_map == STATUS_4)).astype(np.uint8) * 255

    lesion_contour = find_largest_contour(lesion_mask)
    ablation_contour = find_largest_contour(ablation_pattern)

    # # Visualize contours.
    # fig, ax = plt.subplots()
    # ax.add_patch(plt.Polygon(lesion_contour[:, 0, :], fill=False, edgecolor='blue', label='Lesion Contour'))
    # ax.add_patch(plt.Polygon(ablation_contour[:, 0, :], fill=False, edgecolor='red', label='Ablation Contour'))
    # ax.set_xlim(0, width)
    # ax.set_ylim(0, height)
    # ax.legend()
    # plt.savefig("contours.png")

    angles = np.arange(0, 360, 1)  # Angles from 0° to 360°
    radial_diff = []

    for angle in angles:
        radial_dist1 = compute_radial_distance_at_angle(lesion_contour, center, angle)
        radial_dist2 = compute_radial_distance_at_angle(ablation_contour, center, angle)
        
        diff = np.abs(radial_dist1 - radial_dist2)
        radial_diff.append(diff)

    # # plot radial difference.
    # plt.plot(angles, radial_diff)
    # plt.savefig("radial_diff.png")

    avg_radial_diff = np.mean(radial_diff)
    
    return avg_radial_diff


def find_largest_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    return contour


def compute_radial_distance_at_angle(contour, center, angle):
    """
    Compute the radial distance for a given angle by finding the closest point
    on the contour in the direction of the angle.
    """
    # Calculate the unit vector for the given angle (in radians)
    angle_rad = np.deg2rad(angle)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Calculate the vector from the center to each point on the contour
    contour_vectors = contour[:, 0, :] - np.array(center)
    
    # Normalize the contour vectors
    contour_directions = contour_vectors / np.linalg.norm(contour_vectors, axis=1, keepdims=True)
    
    # Find the contour point closest to the desired angle
    cos_similarities = np.dot(contour_directions, direction)
    closest_point_idx = np.argmax(cos_similarities)
    
    # Compute the radial distance to that point
    radial_distance = np.linalg.norm(contour_vectors[closest_point_idx])
    
    return radial_distance


def compute_result_from_status(status_map, weight_ablated_tumor=1, weight_ablated_healthy=-1, weight_nonablated_tumor=0):
    # Consider probe center as ablated
    # status_map[49:52, 49:52] = STATUS_4

    healthy_ablated_idx = status_map == STATUS_2
    tumor_ablated_idx = status_map == STATUS_4
    lesion_mask = (status_map == STATUS_4) | (status_map == STATUS_3)

    tumor_size = np.count_nonzero(lesion_mask)

    ablated_tumor_score = np.count_nonzero(tumor_ablated_idx) / tumor_size * 100
    ablated_healthy_score = np.count_nonzero(healthy_ablated_idx) / tumor_size * 100
    ablated_total = np.count_nonzero(tumor_ablated_idx) + np.count_nonzero(healthy_ablated_idx)
    nonablated_tumor_score = tumor_size - np.count_nonzero(tumor_ablated_idx)

    # Calculate the total score as a weighted sum
    total_score = weight_ablated_tumor * ablated_tumor_score + \
                  weight_ablated_healthy * ablated_healthy_score + \
                  weight_nonablated_tumor * nonablated_tumor_score
    
    avg_radial_diff = compute_radial_diff(status_map)
    
    return total_score, ablated_tumor_score, ablated_healthy_score, avg_radial_diff

