import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


def load_image_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ECC-Based Alignment

def align_images_ecc(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-8)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            img1_gray, img2_gray, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria
        )
    except cv2.error as e:
        print(f"ECC alignment failed: {e}")
        return None

    aligned_img2 = cv2.warpPerspective(
        img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    return aligned_img2


def show_comparison(img1, img2_unaligned, img2_aligned):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Reference Image (a1.png)")
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Before Alignment (a2.png)")
    plt.imshow(img2_unaligned)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("After Alignment")
    plt.imshow(img2_aligned)
    plt.axis('off')

    plt.tight_layout()
    plt.show()




def normalize_environment(img_reference, img_to_adjust):
    normalized = match_histograms(img_to_adjust, img_reference, channel_axis=-1)
    return normalized

def show_normalization_comparison(reference, before, after):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Reference (a1.png)")
    plt.imshow(reference)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Aligned (Before Normalization)")
    plt.imshow(before)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("After Histogram Matching")
    plt.imshow(after)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def detect_changes(reference, normalized):
    diff = cv2.absdiff(reference, normalized)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_diff, 65, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return clean_mask

def show_change_overlay(reference, change_mask):
    overlay = reference.copy()
    overlay[change_mask > 0] = [255, 0, 0]  # Highlight changes in red

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Binary Change Mask")
    plt.imshow(change_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Changes Highlighted on Reference")
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img1 = load_image_rgb("D:\\indrones\\images\\a1.png")
    img2 = load_image_rgb("D:\\indrones\\images\\a2.png")

    aligned_img2 = align_images_ecc(img1, img2)
    if aligned_img2 is not None:
        aligned_bgr = cv2.cvtColor(aligned_img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite("D:\\indrones\\images\\aligned_a2.png", aligned_bgr)
        print("✅ Aligned image saved as 'aligned_a2.png'")
        show_comparison(img1, img2, aligned_img2)
    else:
        print("❌ Image alignment failed.")
        exit()

    normalized_img2 = normalize_environment(img1, aligned_img2)
    normalized_bgr = cv2.cvtColor(normalized_img2, cv2.COLOR_RGB2BGR)
    cv2.imwrite("D:\\indrones\\images\\normalized_a2.png", normalized_bgr)
    print("✅ Normalized image saved as 'normalized_a2.png'")
    show_normalization_comparison(img1, aligned_img2, normalized_img2)

    change_mask = detect_changes(img1, normalized_img2)
    cv2.imwrite("D:\\indrones\\images\\change_mask.png", change_mask)
    print("✅ Change mask saved as 'change_mask.png'")
    show_change_overlay(img1, change_mask)
