import os
from PIL import Image
import zipfile
from io import BytesIO

def separate_results(processed_images):
    matched = [img for img, is_matched in processed_images if is_matched]
    unmatched = [img for img, is_matched in processed_images if not is_matched]
    return matched, unmatched

def download_results(matched, unmatched):
    with zipfile.ZipFile('results.zip', 'w') as zipf:
        for i, img in enumerate(matched):
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            zipf.writestr(f'matched/matched_{i+1}.png', img_byte_arr.getvalue())
        
        for i, img in enumerate(unmatched):
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            zipf.writestr(f'unmatched/unmatched_{i+1}.png', img_byte_arr.getvalue())
    
    return 'results.zip'
