import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

def choose_image():
    print("\n1. Use image from 'assets'\n2. Upload new image")
    try:
        opt = int(input("Choose option: "))
    except ValueError:
        print("Invalid input.")
        return None

    if opt == 1:
        os.makedirs("assets", exist_ok=True)
        files = [f for f in os.listdir("assets") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            print("No images found in 'assets' folder.")
            return None
        for i, f in enumerate(files, 1):
            print(f"{i}. {f}")
        try:
            idx = int(input("Select image number: ")) - 1
            return os.path.join("assets", files[idx]) if 0 <= idx < len(files) else None
        except ValueError:
            return None
    elif opt == 2:
        root = Tk(); root.withdraw(); root.attributes('-topmost', True)
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        root.destroy()
        return path if path else None
    return None

def image_smoothing():
    path = choose_image()
    if not path: return
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    filters = {
        1: ("Average", cv2.blur(img, (5, 5))),
        2: ("Gaussian", cv2.GaussianBlur(img, (5, 5), 0)),
        3: ("Median", cv2.medianBlur(img, 5)),
        4: ("Bilateral", cv2.bilateralFilter(img, 9, 75, 75))
    }

    print("\n1.Average  2.Gaussian  3.Median  4.Bilateral  5.All")
    try:
        choice = int(input("Choose filter: "))
    except ValueError:
        print("Invalid input."); return

    if choice == 5:
        plt.figure(figsize=(12, 8))
        titles = ["Original"] + [v[0] for v in filters.values()]
        imgs = [img_rgb] + [v[1] for v in filters.values()]
        for i, (title, im) in enumerate(zip(titles, imgs), 1):
            plt.subplot(2, 3, i); plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.title(title); plt.axis('off')
    elif choice in filters:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("Original"); plt.axis('off')
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(filters[choice][1], cv2.COLOR_BGR2RGB))
        plt.title(filters[choice][0]); plt.axis('off')
    else:
        print("Invalid choice."); return
    plt.show()


def motion_estimation():
    folder = "video_frames_output"
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found!"); return

    frames = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
    if len(frames) < 2:
        print("Need at least 2 frames."); return

    for i, f in enumerate(frames, 1): print(f"{i}. {f}")
    try:
        f1, f2 = int(input("First frame: ")) - 1, int(input("Second frame: ")) - 1
        if not (0 <= f1 < len(frames) and 0 <= f2 < len(frames)): return
    except ValueError:
        print("Invalid input."); return

    img1 = cv2.imread(os.path.join(folder, frames[f1]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, frames[f2]), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: print("Error loading frames."); return

    pts1 = cv2.goodFeaturesToTrack(img1, 100, 0.01, 10)
    pts2, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1, None)
    good_old, good_new = pts1[st == 1], pts2[st == 1]
    M, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    print("\nAffine Transformation Matrix:\n", M)

    img_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for n, o in zip(good_new, good_old):
        x1, y1 = map(int, o.ravel()); x2, y2 = map(int, n.ravel())
        cv2.arrowedLine(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

    # Before–After display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img1, cmap='gray'); plt.title("Before"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img_vis[..., ::-1]); plt.title("Motion Vectors (After)"); plt.axis('off')
    plt.show()


while True:
    print("\n==== MINI PROJECT MENU ====")
    print("1. Image Smoothing\n2. Motion Estimation\n3. Exit")
    try:
        c = int(input("Enter choice: "))
    except ValueError:
        print("Invalid input."); continue

    if c == 1: image_smoothing()
    elif c == 2: motion_estimation()
    elif c == 3: print("Goodbye!"); break
    else: print("Invalid choice.")
