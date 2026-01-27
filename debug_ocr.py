# -*- coding: utf-8 -*-
"""OCR ciktisini debug et"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["OMP_NUM_THREADS"] = "1"
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang='en',
    det_db_thresh=0.25,
    det_db_box_thresh=0.6,
    det_db_unclip_ratio=1.75,
    use_angle_cls=True,
    use_gpu=False,
    show_log=False
)

# Test fisleri
test_files = ['o2.jpeg', 'o3.png', 'o18.png', 'o19.jpeg']

for f in test_files:
    path = f"data/{f}"
    if os.path.exists(path):
        print(f"\n{'='*50}")
        print(f"DOSYA: {f}")
        print('='*50)
        result = ocr.ocr(path, cls=True)
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                # KDV, TVA, TOPLAM iceren satirlari vurgula
                if any(kw in text.upper() for kw in ['KDV', 'TVA', 'TOPLAM', 'TOTAL', 'TOP']):
                    print(f">>> {text} ({conf:.2f})")
                else:
                    print(f"    {text} ({conf:.2f})")
