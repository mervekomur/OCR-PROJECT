# Ablation Study & Ensemble Benchmark Raporu

**Tarih:** 2026-01-28 15:13
**Toplam Dosya:** 27

## Preprocessing Adimlari

| Adim | Aciklama |
|------|----------|
| original | Hicbir islem yok |
| grayscale | Sadece gri tonlama |
| binary_otsu | Otsu Thresholding |
| adaptive_thresh | Adaptive Gaussian Thresholding |
| denoise | Fast NL Means Denoising |

## Yontem Bazli Sonuclar (Toplam Alani)

| Yontem | Dogru | Toplam | Basari |
|--------|-------|--------|--------|
| paddle_original | 11 | 27 | %40.7 |
| paddle_grayscale | 11 | 27 | %40.7 |
| **ensemble** | 11 | 27 | %40.7 |
| paddle_binary_otsu | 10 | 27 | %37.0 |
| paddle_denoise | 10 | 27 | %37.0 |
| paddle_adaptive_thresh | 4 | 27 | %14.8 |
| easy_original | 0 | 27 | %0.0 |
| easy_grayscale | 0 | 27 | %0.0 |
| easy_binary_otsu | 0 | 27 | %0.0 |
| easy_adaptive_thresh | 0 | 27 | %0.0 |
| easy_denoise | 0 | 27 | %0.0 |

## Preprocessing Etkisi (PaddleOCR)

| Preprocessing | Dogru | Fark (vs Original) |
|---------------|-------|---------------------|
| original | 11 | 0 |
| grayscale | 11 | 0 |
| binary_otsu | 10 | -1 |
| adaptive_thresh | 4 | -7 |
| denoise | 10 | -1 |

## Preprocessing Etkisi (EasyOCR)

| Preprocessing | Dogru | Fark (vs Original) |
|---------------|-------|---------------------|
| original | 0 | 0 |
| grayscale | 0 | 0 |
| binary_otsu | 0 | 0 |
| adaptive_thresh | 0 | 0 |
| denoise | 0 | 0 |

## Ensemble Voting Analizi

- **Ensemble Basari:** %40.7
- **En Iyi Tek Yontem:** paddle_original (11 dogru)
- **Ensemble:** Tek yontemle ayni performans