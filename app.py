import cv2
import numpy as np
import argparse

def run_fast_rcnn_demo(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Fast R-CNN (Simplified Detection)', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_motion_estimation(frame1_path, frame2_path):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    if frame1 is None or frame2 is None:
        print("Error: Cannot read frames.")
        return

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    pts1 = cv2.goodFeaturesToTrack(gray1, maxCorners=500, qualityLevel=0.01, minDistance=10)
    if pts1 is None:
        print("Warning: No features found in frame1 to track")
        return

    pts2, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, pts1, None)
    if pts2 is None or st is None:
        print("Warning: Optical flow failed")
        return

    st = st.reshape(-1)
    good1 = pts1[st == 1]
    good2 = pts2[st == 1]

    if len(good1) < 3:
        print("Not enough good matches for affine estimation")
        return

    matrix, inliers = cv2.estimateAffinePartial2D(good1, good2)
    print("Estimated Affine Matrix:", matrix)

    if inliers is not None:
        mask = inliers.reshape(-1) == 1
    else:
        mask = np.ones(len(good1), dtype=bool)

    for p1, p2, inl in zip(good1, good2, mask):
        if not inl:
            continue
        x1, y1 = p1.ravel()
        x2, y2 = p2.ravel()
        cv2.arrowedLine(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, tipLength=0.3)

    cv2.imshow('Motion Estimation (Affine)', frame2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fastrcnn', 'motion'], help='Choose mode: fastrcnn or motion')
    parser.add_argument('--image', help='Image path for fastrcnn mode')
    parser.add_argument('--frame1', help='First frame for motion mode')
    parser.add_argument('--frame2', help='Second frame for motion mode')

    args = parser.parse_args()

    # To use your own input:
    # For single image detection: --mode fastrcnn --image path/to/your/image.jpg
    # For motion estimation (e.g., from video): --mode motion --frame1 path/to/frame1.jpg --frame2 path/to/frame2.jpg

    if args.mode == 'fastrcnn':
        if not args.image:
            print('Error: Please provide --image for fastrcnn mode')
            return
        run_fast_rcnn_demo(args.image)

    elif args.mode == 'motion':
        if not args.frame1 or not args.frame2:
            print('Error: Please provide --frame1 and --frame2 for motion mode')
            return
        run_motion_estimation(args.frame1, args.frame2)

if __name__ == '__main__':
    main()