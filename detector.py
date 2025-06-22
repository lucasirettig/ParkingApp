from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from preprocessor import preprocess_image_for_detection

# Load model once
model = YOLO("models/yolov8n.pt")  # adjust path as needed

def detect_objects(image_path, conf=0.1, iou=0.4):
    """
    Run detection on two differently preprocessed versions:
    - Enhanced (CLAHE + bright gamma)
    - Low-contrast (darkening gamma)
    Merge boxes via NMS and return list of detections.
    """
    original = cv2.imread(image_path)
    # First preprocessing: enhance dark cars
    enhanced = preprocess_image_for_detection(original.copy())
    # Second preprocessing: reduce contrast for white cars
    # apply inverse gamma <1
    low_contrast = original.copy()
    gamma = 0.7  # darken highlights
    inv_lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0,256)]).astype('uint8')
    low_contrast = cv2.LUT(low_contrast, inv_lut)

    # Run model on both images
    res_enh = model(enhanced, conf=conf, iou=iou)[0]
    res_low = model(low_contrast, conf=conf, iou=iou)[0]

    # Combine boxes
    all_boxes = []
    for r in (res_enh, res_low):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf_score = float(box.conf[0])
            cls = int(box.cls[0])
            all_boxes.append((x1, y1, x2, y2, conf_score, cls))

    # Apply NMS
    picked = []
    boxes = sorted(all_boxes, key=lambda x: x[4], reverse=True)
    while boxes:
        bx = boxes.pop(0)
        picked.append(bx)
        keep = []
        for b in boxes:
            xx1 = max(bx[0], b[0]); yy1 = max(bx[1], b[1])
            xx2 = min(bx[2], b[2]); yy2 = min(bx[3], b[3])
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            inter = w * h
            area1 = (bx[2]-bx[0])*(bx[3]-bx[1])
            area2 = (b[2]-b[0])*(b[3]-b[1])
            iou_val = inter / (area1 + area2 - inter + 1e-9)
            if iou_val < iou:
                keep.append(b)
        boxes = keep

    detections = []
    # Draw final boxes on enhanced view
    for x1, y1, x2, y2, conf_score, cls in picked:
        label = model.names[cls]
        detections.append({
            'class': label,
            'confidence': round(conf_score, 2),
            'box': [int(x1), int(y1), int(x2), int(y2)]
        })
        # annotate for display
        cv2.rectangle(enhanced, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(enhanced, f"{label} {conf_score:.2f}",
                    (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

    # Display results
    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enh_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,12))
    ax1.imshow(orig_rgb); ax1.set_title("Original"); ax1.axis('off')
    ax2.imshow(enh_rgb); ax2.set_title("Detection (Enhanced + Low-contrast)"); ax2.axis('off')
    plt.tight_layout(); plt.show()

    return detections