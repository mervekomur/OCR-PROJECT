# OCR Benchmark Raporu

**Tarih:** 2026-01-28 11:45
**Toplam Dosya:** 27

## 1. Genel Performans Özeti

| Metrik | Doğru | Toplam | Başarı | Atlanan |
|--------|-------|--------|--------|---------|
| Tarih | 15 | 25 | %60.0 | 2 |
| Toplam | 17 | 27 | %63.0 | 0 |
| KDV | 8 | 22 | %36.4 | 5 |
| Vergi No | 10 | 21 | %47.6 | 6 |
| **GENEL** | **50** | **95** | **%52.6** | - |

## 2. Kategori Bazlı Performans

| Kategori | Dosya | Tarih | Toplam | KDV | Vergi No |
|----------|-------|-------|--------|-----|----------|
| TR Fişleri | 10 | 6/10 (%60) | 9/10 (%90) | 8/9 (%89) | 3/9 (%33) |
| FR Fişleri | 11 | 6/9 (%67) | 8/11 (%73) | 0/9 (%0) | 5/6 (%83) |
| Makbuzlar | 6 | 3/6 (%50) | 0/6 (%0) | 0/4 (%0) | 2/6 (%33) |

## 3. OCR Motor Karşılaştırması

| Motor | Tarih | Toplam | KDV | Vergi No | Genel |
|-------|-------|--------|-----|----------|-------|
| PaddleOCR | 15/25 (%60) | 17/27 (%63) | 8/22 (%36) | 10/21 (%48) | %52.6 |
| Google Vision | - | - | - | - | Test edilmedi |

## 4. Dosya Bazlı Detaylı Sonuçlar

| Dosya | Market | Tarih | Toplam | KDV | Vergi No | Hata |
|-------|--------|-------|--------|-----|----------|------|
| makbuz1.png | Halil Nalcakan | ✗ | ✗ | - | ✗ | 3 |
| makbuz2.png | Europcar | ✓ | ✗ | ✗ | ✗ | 3 |
| makbuz3.png | OHEM ENERJI URUNLERI LTD  | ✓ | ✗ | ✗ | ✗ | 3 |
| makbuz4.png | Ismont | ✓ | ✗ | ✗ | ✗ | 3 |
| makbuz5.png | Bastruk Brahim / Traser T | ✗ | ✗ | ✗ | ✓ | 3 |
| makbuz7.jpeg | La Ricetta SDQ EIRL (Linc | ✗ | ✗ | - | ✓ | 2 |
| o-1.png | Yalova Ro-Ro Terminali An | ✗ | ✓ | ✗ | ✗ | 3 |
| o-2.png | Lidl | ✗ | ✓ | ✗ | ✓ | 2 |
| o-3.jpeg | Fat Sultan Sofrasi Lokant | ✓ | ✓ | ✓ | ✓ | 0 |
| o1.png | Mehmetcik Vakfi Turizm Pe | ✓ | ✗ | - | ✗ | 2 |
| o10.png | Lidl | ✗ | ✓ | ✗ | ✓ | 2 |
| o11.png | Cora Wittenheim | - | ✓ | ✗ | - | 1 |
| o12.png | Leroy Merlin Morschwiller | - | ✓ | ✗ | - | 1 |
| o13.png | Epicerie Parsa | ✓ | ✗ | ✗ | ✓ | 2 |
| o14.png | Stationnement (Park Fisi) | ✓ | ✗ | - | - | 1 |
| o15.png | Palmiye Resto | ✓ | ✓ | ✗ | ✓ | 1 |
| o16.png | Auchan Ensisheim | ✗ | ✓ | ✗ | - | 2 |
| o17.png | Cora Wittenheim (Station) | ✓ | ✓ | ✗ | - | 1 |
| o18.png | Starbucks Coffee | ✓ | ✓ | ✓ | ✗ | 1 |
| o19.jpeg | Ispark Istanbul Otopark I | ✓ | ✓ | ✓ | ✗ | 1 |
| o2.jpeg | Otoyol Yatirim Isletme A. | ✓ | ✓ | ✓ | ✓ | 0 |
| o3.png | Lotus Cafe (Mahmut Canpol | ✓ | ✓ | ✓ | ✓ | 0 |
| o4.png | Intera Gida Ins. ve Tic.  | ✗ | ✓ | ✓ | ✗ | 2 |
| o6.png | Doganli Gida (Muhammet Al | ✗ | ✓ | ✓ | - | 1 |
| o7.png | Kirisoglu Petrol Kimya Sa | ✗ | ✓ | ✓ | ✗ | 2 |
| o8.png | Kebab House | ✓ | ✓ | - | ✗ | 1 |
| o9.png | McDonald's Mulhouse Porte | ✓ | ✗ | ✗ | ✓ | 2 |

## 5. Başarısız Sonuçların Detayı

### Tarih Hataları (10 adet)

| Dosya | Beklenen (GT) | OCR Sonucu |
|-------|---------------|------------|
| makbuz1.png | 19/06/2024 | - |
| makbuz5.png | 2024-07-15 | 19/06/2024 |
| makbuz7.jpeg | 14/12/2021 | - |
| o4.png | 17/06/2024 | 19/06/2024 |
| o6.png | 17/06/2024 | 12/06/2024 |
| o7.png | 18/07/2021 | 18/06/2024 |
| o10.png | 19/06/2024 | - |
| o16.png | 24/06/2024 | - |
| o-1.png | 12/06/2024 | - |
| o-2.png | 19/06/2024 | - |

### Toplam Hataları (10 adet)

| Dosya | Beklenen (GT) | OCR Sonucu |
|-------|---------------|------------|
| makbuz1.png | 980.0 | 21.12 |
| makbuz2.png | 1307.58 | 1525.75 |
| makbuz3.png | 2520 | 520.0 |
| makbuz4.png | 2099.99 | 66.13 |
| makbuz5.png | 1523.94 | 11343.94 |
| makbuz7.jpeg | 2450.0 | - |
| o1.png | 1000.0 | 1.0 |
| o9.png | 43.30 | 1.0 |
| o13.png | 63.96 | 15.99 |
| o14.png | 1.0 | - |

### KDV Hataları (14 adet)

| Dosya | Beklenen (GT) | OCR Sonucu |
|-------|---------------|------------|
| makbuz2.png | 97.98 | - |
| makbuz3.png | 420 | - |
| makbuz4.png | 190.91 | - |
| makbuz5.png | - | - |
| o9.png | 3.94 | - |
| o10.png | 4.59 | - |
| o11.png | 0.66 | - |
| o12.png | null | - |
| o13.png | 3.33 | - |
| o15.png | 4.09 | - |
| o16.png | 2.84 | - |
| o17.png | 13.63 | - |
| o-1.png | 310.0 | 340.0 |
| o-2.png | 4.59 | - |

### Vergi No Hataları (11 adet)

| Dosya | Beklenen (GT) | OCR Sonucu |
|-------|---------------|------------|
| makbuz1.png | 14654556332 | - |
| makbuz2.png | 106.042.061 | - |
| makbuz3.png | 8591406336 | 6360371836 |
| makbuz4.png | 8591406336 | 4820398418 |
| o1.png | 6150768602 | 8150166942 |
| o4.png | 1150404001 | - |
| o7.png | 0773402W670 | 7340209267 |
| o8.png | W305910318 | 85020438900017 |
| o18.png | 7004310115 | - |
| o19.jpeg | 5260047973 | - |
| o-1.png | 4890027464 | - |

## 6. En Sorunlu Dosyalar (Top 10)

| Sıra | Dosya | Hata Sayısı | Market |
|------|-------|-------------|--------|
| 1 | makbuz1.png | 3 | Halil Nalcakan |
| 2 | makbuz2.png | 3 | Europcar |
| 3 | makbuz3.png | 3 | OHEM ENERJI URUNLERI LTD STI |
| 4 | makbuz4.png | 3 | Ismont |
| 5 | makbuz5.png | 3 | Bastruk Brahim / Traser Trafo  |
| 6 | o-1.png | 3 | Yalova Ro-Ro Terminali Anonim  |
| 7 | makbuz7.jpeg | 2 | La Ricetta SDQ EIRL (Lincoln S |
| 8 | o1.png | 2 | Mehmetcik Vakfi Turizm Petrol  |
| 9 | o4.png | 2 | Intera Gida Ins. ve Tic. Ltd.  |
| 10 | o7.png | 2 | Kirisoglu Petrol Kimya San. ve |

## 7. İyileştirme Önerileri

- [ ] **KDV Parser İyileştirmesi**: TVA pattern'leri genişletilmeli (FR fişleri için)
- [ ] **Vergi No Parser İyileştirmesi**: V.D. pattern'leri düzeltilmeli
- [ ] **Makbuz Preprocessing**: Gölge kaldırma ve kırpma eklenmeli
- [ ] **Google Vision OCR**: Karşılaştırma için test edilmeli
- [ ] **Ensemble Yaklaşım**: Birden fazla OCR motoru birleştirilebilir
