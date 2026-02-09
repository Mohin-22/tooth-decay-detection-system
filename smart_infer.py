# smart_infer.py
import torch
import numpy as np
import cv2
from PIL import Image
from main import FasterRCNNLightning
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_MODEL = None

def _load_model(ckpt_path: str):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _MODEL = FasterRCNNLightning.load_from_checkpoint(ckpt_path)
    _MODEL.to(device)
    _MODEL.eval()
    return _MODEL

def smart_decay_infer(image: np.ndarray, ckpt_path: str = '3.ckpt', threshold: float = 0.3):
    """
    Simple inference that shows ALL detections (for now)
    """
    model = _load_model(ckpt_path)
    
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image
    
    # Resize for model
    original_h, original_w = image_rgb.shape[:2]
    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert('RGB')
    pil_resized = pil.resize((512, 512), resample=Image.BILINEAR)
    arr = np.array(pil_resized).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model([input_tensor])
    
    if not outputs or len(outputs) == 0:
        # No detections - return original image
        result_vis = image_rgb.copy()
        cv2.putText(result_vis, "No detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return result_vis, 0.0, "No dental issues detected"
    
    preds = outputs[0]
    
    # Filter by confidence
    if len(preds['scores']) > 0:
        mask = preds['scores'] > threshold
        scores = preds['scores'][mask]
        labels = preds['labels'][mask]
        boxes = preds['boxes'][mask]
        
        print(f"Found {len(scores)} detections after filtering")
        
        # Scale boxes to original image size
        scaled_boxes = []
        for box in boxes.cpu().numpy():
            x1, y1, x2, y2 = box
            x1 = int(x1 * original_w / 512)
            y1 = int(y1 * original_h / 512) 
            x2 = int(x2 * original_w / 512)
            y2 = int(y2 * original_h / 512)
            scaled_boxes.append([x1, y1, x2, y2])
        
        # Create visualization on original image
        result_vis = image_rgb.copy()
        
        # Draw all detections
        for i, (box, score, label) in enumerate(zip(scaled_boxes, scores, labels)):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            color = (0, 0, 255)  # Red for all detections
            cv2.rectangle(result_vis, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"Det_{i+1}(L{label})"
            cv2.putText(result_vis, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Calculate area percentage
        decay_percent = calculate_decay_percent(scaled_boxes, image_rgb.shape)
        advice = get_advice(decay_percent, len(scaled_boxes))
        
        # Add summary text
        summary = f"Detections: {len(scaled_boxes)}"
        cv2.putText(result_vis, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return result_vis, decay_percent, advice
    
    # No confident detections
    result_vis = image_rgb.copy()
    cv2.putText(result_vis, "No confident detections", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return result_vis, 0.0, "No significant findings"

def calculate_decay_percent(boxes, img_shape):
    if not boxes:
        return 0.0
    total_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        total_area += area
    img_area = img_shape[0] * img_shape[1]
    return (total_area / img_area) * 100

def get_advice(decay_percent, num_detections):
    if num_detections == 0:
        return "✅ No significant dental issues detected. Maintain good oral hygiene!"
    elif decay_percent < 2:
        return "🟡 Minor findings detected. Regular dental check-up recommended."
    elif decay_percent < 5:
        return "🟠 Moderate findings. Dental consultation advised."
    else:
        return "🔴 Significant findings detected. Please consult a dentist."