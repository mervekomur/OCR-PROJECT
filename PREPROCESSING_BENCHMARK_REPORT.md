# Preprocessing Benchmark Raporu

**Tarih:** 2026-01-28 14:59
**OCR Motoru:** PaddleOCR
**Dosya Sayisi:** 27

## Preprocessing Adimlari

| Kod | Aciklama |
|-----|----------|
| raw | Ham (Islenmemis) |
| grayscale | Grayscale |
| otsu | Otsu Threshold |
| adaptive | Adaptive Threshold |
| shadow | Golge Kaldirma |
| denoise | Gurultu Azaltma |
| sharpen | Keskinlestirme |
| shadow_otsu | Golge + Otsu |
| all_clean | Full Clean |

## Sonuc Tablosu

| Preprocessing | Tarih | Toplam | KDV | Vergi No | **TOTAL** |
|---------------|-------|--------|-----|----------|-----------|
| raw | 15 | 17 | 8 | 10 | **50** |
| grayscale | 15 | 17 | 8 | 10 | **50** |
| shadow | 15 | 16 | 8 | 6 | **45** |
| sharpen | 12 | 18 | 8 | 6 | **44** |
| denoise | 13 | 17 | 7 | 6 | **43** |
| adaptive | 13 | 19 | 5 | 4 | **41** |
| shadow_otsu | 9 | 18 | 7 | 3 | **37** |
| otsu | 8 | 17 | 7 | 4 | **36** |
| all_clean | 6 | 14 | 5 | 2 | **27** |

## En Iyi Kombinasyon: **raw** (Ham (Islenmemis))