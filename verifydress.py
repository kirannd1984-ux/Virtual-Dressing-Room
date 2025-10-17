# verify_dress_template.py
import cv2
import os

def verify_professional_template(image_path):
    """Verify dress template meets professional specifications"""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return False
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return False
    
    h, w = img.shape[:2]
    has_alpha = img.shape[2] == 4
    
    print(f"✅ File: {image_path}")
    print(f"✅ Dimensions: {w} x {h}")
    print(f"✅ Target: 400 x 800")
    print(f"✅ Aspect Ratio: 1:{h/w:.2f} (Target: 1:2.00)")
    print(f"✅ Has Alpha: {'Yes' if has_alpha else 'No'}")
    
    # Check if close to target dimensions
    if abs(w - 400) <= 20 and abs(h - 800) <= 40:
        print("✅ Dimensions: PERFECT for virtual try-on")
    else:
        print("⚠️  Dimensions: May need resizing")
    
    # Check transparency
    if has_alpha:
        alpha_channel = img[:, :, 3]
        transparent_pixels = (alpha_channel == 0).sum()
        total_pixels = alpha_channel.size
        transparency_pct = (transparent_pixels / total_pixels) * 100
        print(f"✅ Transparency: {transparency_pct:.1f}% transparent pixels")
        
        if transparency_pct > 5:
            print("✅ Has proper transparent areas (neck/background)")
        else:
            print("⚠️  May lack sufficient transparency")
    
    return True

# Verify all dresses in folder
if __name__ == "__main__":
    dress_folder = "dresses"
    if os.path.exists(dress_folder):
        for file in os.listdir(dress_folder):
            if file.lower().endswith('.png'):
                print(f"\n{'='*50}")
                verify_professional_template(os.path.join(dress_folder, file))
    else:
        print("No 'dresses' folder found")