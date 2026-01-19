# -*- coding: utf-8 -*-
"""
SAM (Segment Anything Model) tabanlı fiş/belge tespiti

Kullanım:
    from src.sam_detector import ReceiptDetector

    detector = ReceiptDetector()
    result = detector.detect("path/to/image.jpg")

    # result içeriği:
    # - mask: binary maske
    # - contour: kontur noktaları
    # - corners: 4 köşe koordinatı
    # - cropped: kesilmiş görüntü
"""

import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class ReceiptDetector:
    """SAM tabanlı fiş/belge tespit sınıfı"""

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: SAM model dosyasının yolu (.pth)
            device: 'cuda' veya 'cpu' (None ise otomatik seçilir)
        """
        if model_path is None:
            # Varsayılan model yolu
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, 'sam_vit_b.pth')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SAM model dosyası bulunamadı: {model_path}\n"
                "Model indirmek için: "
                "curl -L -o sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._model = None
        self._mask_generator = None

    def _load_model(self):
        """Model lazy loading"""
        if self._model is None:
            self._model = sam_model_registry["vit_b"](checkpoint=self.model_path)
            self._model.to(self.device)

            self._mask_generator = SamAutomaticMaskGenerator(
                model=self._model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                min_mask_region_area=1000,
            )

    def detect(self, image_or_path):
        """
        Görüntüden fiş/belge tespit et

        Args:
            image_or_path: numpy array (BGR) veya dosya yolu

        Returns:
            dict: {
                'success': bool,
                'mask': binary maske,
                'contour': kontur noktaları,
                'corners': 4 köşe koordinatı,
                'bbox': bounding box (x, y, w, h),
                'cropped': kesilmiş görüntü,
                'area': maske alanı,
                'num_masks': toplam bulunan maske sayısı
            }
        """
        self._load_model()

        # Görüntüyü yükle
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                return {'success': False, 'error': f'Görüntü yüklenemedi: {image_or_path}'}
        else:
            image = image_or_path

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Maskeleri üret
        masks = self._mask_generator.generate(image_rgb)

        # En uygun maskeyi bul
        best_mask = self._find_best_mask(masks, image.shape)

        if best_mask is None:
            return {
                'success': False,
                'error': 'Uygun maske bulunamadı',
                'num_masks': len(masks)
            }

        # Sonuçları hazırla
        mask = best_mask['segmentation']
        contour = self._mask_to_contour(mask)
        corners = self._contour_to_quad(contour) if contour is not None else None
        bbox = cv2.boundingRect(contour) if contour is not None else None
        cropped = self._crop_image(image, contour) if contour is not None else None

        return {
            'success': True,
            'mask': mask,
            'contour': contour,
            'corners': corners,
            'bbox': bbox,
            'cropped': cropped,
            'area': best_mask['area'],
            'num_masks': len(masks)
        }

    def detect_and_save(self, image_path, output_dir=None):
        """
        Tespit et ve sonuçları kaydet

        Args:
            image_path: görüntü dosya yolu
            output_dir: çıktı klasörü (None ise görüntü ile aynı klasör)

        Returns:
            dict: detect() çıktısı + kaydedilen dosya yolları
        """
        result = self.detect(image_path)

        if not result['success']:
            return result

        # Çıktı klasörü
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)

        # Dosya adı
        basename = os.path.splitext(os.path.basename(image_path))[0]

        # Orijinal görüntü
        image = cv2.imread(image_path)

        # Sonuç görüntüsü (overlay + kontur)
        result_img = self._draw_result(image, result)
        result_path = os.path.join(output_dir, f'{basename}_sam_result.jpg')
        cv2.imwrite(result_path, result_img)

        # Kesilmiş görüntü
        if result['cropped'] is not None:
            cropped_path = os.path.join(output_dir, f'{basename}_sam_cropped.jpg')
            cv2.imwrite(cropped_path, result['cropped'])
            result['cropped_path'] = cropped_path

        # Maske
        mask_path = os.path.join(output_dir, f'{basename}_sam_mask.jpg')
        cv2.imwrite(mask_path, (result['mask'] * 255).astype(np.uint8))

        result['result_path'] = result_path
        result['mask_path'] = mask_path

        return result

    def _find_best_mask(self, masks, image_shape):
        """En uygun fiş maskesini bul"""
        h, w = image_shape[:2]
        img_area = h * w
        center = (w // 2, h // 2)

        best_mask = None
        best_score = 0

        for mask_data in masks:
            area = mask_data['area']
            bbox = mask_data['bbox']

            # Alan filtresi: %5 ile %90 arası
            area_ratio = area / img_area
            if area_ratio < 0.05 or area_ratio > 0.90:
                continue

            # Merkeze yakınlık
            mask_center_x = bbox[0] + bbox[2] // 2
            mask_center_y = bbox[1] + bbox[3] // 2
            dist_to_center = np.sqrt((mask_center_x - center[0])**2 + (mask_center_y - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            center_score = 1 - (dist_to_center / max_dist)

            # Stability score
            stability = mask_data.get('stability_score', 0.5)

            # Toplam skor
            score = area_ratio * 0.4 + center_score * 0.4 + stability * 0.2

            if score > best_score:
                best_score = score
                best_mask = mask_data

        return best_mask

    def _mask_to_contour(self, mask):
        """Binary maskeden kontur çıkar"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        return max(contours, key=cv2.contourArea)

    def _contour_to_quad(self, contour):
        """Konturu 4 köşeli dörtgene dönüştür"""
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return np.int32(box)

    def _crop_image(self, image, contour):
        """Kontur bounding box'ına göre görüntüyü kes"""
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]

    def _draw_result(self, image, result):
        """Sonucu görselleştir"""
        output = image.copy()

        # Yeşil overlay
        if result.get('mask') is not None:
            overlay = output.copy()
            overlay[result['mask']] = [0, 255, 0]
            output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

        # Kırmızı kontur
        if result.get('contour') is not None:
            cv2.drawContours(output, [result['contour']], -1, (0, 0, 255), 3)

        # Mavi köşeler
        if result.get('corners') is not None:
            for (x, y) in result['corners']:
                cv2.circle(output, (int(x), int(y)), 8, (255, 0, 0), -1)

        return output


# Convenience fonksiyonları
_detector = None

def get_detector():
    """Singleton detector instance"""
    global _detector
    if _detector is None:
        _detector = ReceiptDetector()
    return _detector


def detect_receipt(image_or_path):
    """Hızlı tespit fonksiyonu"""
    return get_detector().detect(image_or_path)


def detect_and_save(image_path, output_dir=None):
    """Hızlı tespit ve kaydetme fonksiyonu"""
    return get_detector().detect_and_save(image_path, output_dir)
