# OCR Benchmark Matrix Raporu

**Tarih:** 2026-01-28
**Hazırlayan:** OCR Pipeline Ekibi
**Toplam Test Dosyası:** 27

---

## 1. Yönetici Özeti

Bu rapor, fatura/fiş görüntülerinden veri çıkarma (OCR) sisteminin performansını değerlendirmektedir. Test edilen alanlar: **Tarih**, **Toplam Tutar**, **KDV** ve **Vergi Numarası**.

### En İyi Kombinasyon: `PaddleOCR + Raw` (%44.2 genel başarı)

---

## 2. Preprocessing Pipeline Karşılaştırması

| Pipeline | Tarih | Toplam | KDV | Vergi No | **GENEL** |
|----------|-------|--------|-----|----------|-----------|
| **Raw (İşlemsiz)** | 15/25 (%60) | 11/27 (%41) | 8/22 (%36) | 8/21 (%38) | **%44.2** |
| Smart (Akıllı) | 14/25 (%56) | 10/27 (%37) | 6/22 (%27) | 4/21 (%19) | %35.8 |

> **Bulgu:** Raw preprocessing, Smart pipeline'dan daha iyi performans gösterdi. Smart pipeline'ın ağır temizlik modu bazı yazıları bozuyor olabilir.

---

## 3. Kategori Bazlı Performans

| Kategori | Dosya Sayısı | Tarih | Toplam | KDV | Vergi No | Ortalama |
|----------|--------------|-------|--------|-----|----------|----------|
| **TR Fişleri** | 11 | %73 | %55 | %78 | %36 | **%60** |
| **FR Fişleri** | 10 | %60 | %40 | %0 | %50 | **%38** |
| **Makbuzlar** | 6 | %50 | %0 | %0 | %17 | **%17** |

---

## 4. Alan Bazlı Detaylı Analiz

### 4.1 Tarih Tespiti (%60 Başarı)

| Durum | Sayı | Açıklama |
|-------|------|----------|
| ✅ Doğru | 15 | Tarih doğru tespit edildi |
| ❌ Yanlış | 10 | Tarih bulunamadı veya hatalı |
| ⏭️ Atlandı | 2 | Ground truth'da tarih yok |

**Sorunlu Durumlar:**
- Gölgeli/bulanık fişlerde tarih okunamıyor
- Farklı tarih formatları (DD/MM/YYYY vs YYYY-MM-DD)

### 4.2 Toplam Tutar Tespiti (%41 Başarı)

| Durum | Sayı | Açıklama |
|-------|------|----------|
| ✅ Doğru | 11 | Tutar doğru tespit edildi |
| ❌ Yanlış | 16 | Tutar bulunamadı veya hatalı |

**Sorunlu Durumlar:**
- Makbuzlarda TOPLAM kelimesi farklı formatlarda
- Binlik ayırıcı (1.000,00 vs 1,000.00) karışıklığı

### 4.3 KDV Tespiti (%36 Başarı)

| Durum | Sayı | Açıklama |
|-------|------|----------|
| ✅ Doğru | 8 | KDV doğru tespit edildi |
| ❌ Yanlış | 14 | KDV bulunamadı veya hatalı |
| ⏭️ Atlandı | 5 | Ground truth'da KDV yok |

**Sorunlu Durumlar:**
- FR fişlerinde TVA pattern'i tanınmıyor (%0 başarı)
- TOPKDV ve değer farklı satırlarda

### 4.4 Vergi Numarası Tespiti (%38 Başarı)

| Durum | Sayı | Açıklama |
|-------|------|----------|
| ✅ Doğru | 8 | Vergi no doğru tespit edildi |
| ❌ Yanlış | 13 | Vergi no bulunamadı veya hatalı |
| ⏭️ Atlandı | 6 | Ground truth'da vergi no yok |

**Sorunlu Durumlar:**
- V.D. pattern'leri çeşitli (KADIKOYV.D., V.D:, V D gibi)
- SIRET numaraları (FR) farklı formatlarda

---

## 5. Dosya Bazlı Detaylı Sonuçlar

### 5.1 Mükemmel Sonuçlar (4/4)

| Dosya | Market | Tarih | Toplam | KDV | Vergi No |
|-------|--------|-------|--------|-----|----------|
| o2.jpeg | Otoyol Yatırım | ✅ | ✅ | ✅ | ✅ |
| o3.png | Lotus Cafe | ✅ | ✅ | ✅ | ✅ |
| o-3.jpeg | Fat Sultan Sofrası | ✅ | ✅ | ✅ | ✅ |

### 5.2 İyi Sonuçlar (3/4)

| Dosya | Market | Tarih | Toplam | KDV | Vergi No |
|-------|--------|-------|--------|-----|----------|
| o18.png | Starbucks Coffee | ✅ | ✅ | ✅ | ❌ |

### 5.3 Orta Sonuçlar (2/4)

| Dosya | Market | Tarih | Toplam | KDV | Vergi No |
|-------|--------|-------|--------|-----|----------|
| o4.png | Intera Gıda | ❌ | ✅ | ✅ | ❌ |
| o6.png | Doğanlı Gıda | ❌ | ✅ | ✅ | ⏭️ |
| o7.png | Kırışoğlu Petrol | ❌ | ✅ | ✅ | ❌ |
| o13.png | Epicerie Parsa | ✅ | ❌ | ❌ | ✅ |
| o15.png | Palmiye Resto | ✅ | ❌ | ❌ | ✅ |
| o17.png | Cora Wittenheim | ✅ | ✅ | ❌ | ⏭️ |
| o19.jpeg | İspark | ✅ | ❌ | ✅ | ❌ |

### 5.4 Başarısız Sonuçlar (0-1/4)

| Dosya | Market | Skor | Ana Sorun |
|-------|--------|------|-----------|
| makbuz1.png | Halil Nalcakan | 0/4 | Görüntü kalitesi düşük |
| makbuz7.jpeg | La Ricetta | 0/4 | Yabancı dil (İspanyolca) |
| o-1.png | Yalova Ro-Ro | 0/4 | Gölgeli görüntü |

---

## 6. Tüm Dosyalar - Detaylı Tablo

| # | Dosya | Market | Kat. | Tarih | Toplam | KDV | Vergi No | Skor |
|---|-------|--------|------|-------|--------|-----|----------|------|
| 1 | makbuz1.png | Halil Nalcakan | Makbuz | ❌ | ❌ | ⏭️ | ❌ | 0/4 |
| 2 | makbuz2.png | Europcar | Makbuz | ✅ | ❌ | ❌ | ❌ | 1/4 |
| 3 | makbuz3.png | OHEM Enerji | Makbuz | ✅ | ❌ | ❌ | ❌ | 1/4 |
| 4 | makbuz4.png | Ismont | Makbuz | ✅ | ❌ | ❌ | ❌ | 1/4 |
| 5 | makbuz5.png | Bastruk Brahim | Makbuz | ❌ | ❌ | ❌ | ✅ | 1/4 |
| 6 | makbuz7.jpeg | La Ricetta | Makbuz | ❌ | ❌ | ⏭️ | ❌ | 0/4 |
| 7 | o1.png | Mehmetçik Vakfı | TR | ✅ | ❌ | ⏭️ | ❌ | 1/4 |
| 8 | o2.jpeg | Otoyol Yatırım | TR | ✅ | ✅ | ✅ | ✅ | **4/4** |
| 9 | o3.png | Lotus Cafe | TR | ✅ | ✅ | ✅ | ✅ | **4/4** |
| 10 | o4.png | Intera Gıda | TR | ❌ | ✅ | ✅ | ❌ | 2/4 |
| 11 | o6.png | Doğanlı Gıda | TR | ❌ | ✅ | ✅ | ⏭️ | 2/4 |
| 12 | o7.png | Kırışoğlu Petrol | TR | ❌ | ✅ | ✅ | ❌ | 2/4 |
| 13 | o8.png | Kebab House | FR | ✅ | ❌ | ⏭️ | ❌ | 1/4 |
| 14 | o9.png | McDonald's | FR | ✅ | ❌ | ❌ | ❌ | 1/4 |
| 15 | o10.png | Lidl | FR | ❌ | ❌ | ❌ | ✅ | 1/4 |
| 16 | o11.png | Cora Wittenheim | FR | ⏭️ | ✅ | ❌ | ⏭️ | 1/4 |
| 17 | o12.png | Leroy Merlin | FR | ⏭️ | ✅ | ❌ | ⏭️ | 1/4 |
| 18 | o13.png | Epicerie Parsa | FR | ✅ | ❌ | ❌ | ✅ | 2/4 |
| 19 | o14.png | Park Fişi | FR | ✅ | ❌ | ⏭️ | ⏭️ | 1/4 |
| 20 | o15.png | Palmiye Resto | FR | ✅ | ❌ | ❌ | ✅ | 2/4 |
| 21 | o16.png | Auchan | FR | ❌ | ✅ | ❌ | ⏭️ | 1/4 |
| 22 | o17.png | Cora Station | FR | ✅ | ✅ | ❌ | ⏭️ | 2/4 |
| 23 | o18.png | Starbucks | TR | ✅ | ✅ | ✅ | ❌ | 3/4 |
| 24 | o19.jpeg | İspark | TR | ✅ | ❌ | ✅ | ❌ | 2/4 |
| 25 | o-1.png | Yalova Ro-Ro | TR | ❌ | ❌ | ❌ | ❌ | 0/4 |
| 26 | o-2.png | Lidl | FR | ❌ | ❌ | ❌ | ✅ | 1/4 |
| 27 | o-3.jpeg | Fat Sultan Sofrası | TR | ✅ | ✅ | ✅ | ✅ | **4/4** |

---

## 7. İyileştirme Önerileri

### 7.1 Kısa Vadeli (Hızlı Kazanımlar)

| Öneri | Etki | Zorluk |
|-------|------|--------|
| TVA pattern'lerini genişlet | KDV %36 → %50+ | Düşük |
| TOPLAM keyword çeşitlerini ekle | Toplam %41 → %55+ | Düşük |
| V.D. pattern'lerini iyileştir | Vergi No %38 → %50+ | Orta |

### 7.2 Orta Vadeli

| Öneri | Etki | Zorluk |
|-------|------|--------|
| Makbuz preprocessing (gölge kaldırma) | Makbuz %17 → %40+ | Orta |
| EasyOCR ensemble | Genel %44 → %55+ | Orta |
| Google Vision test | Potansiyel %60+ | Düşük |

### 7.3 Uzun Vadeli

| Öneri | Etki | Zorluk |
|-------|------|--------|
| Donut/LayoutLM fine-tuning | %70+ potansiyel | Yüksek |
| Çok dilli model eğitimi | Tüm kategorilerde artış | Yüksek |

---

## 8. Sonuç

| Metrik | Değer |
|--------|-------|
| **Genel Başarı Oranı** | %44.2 |
| **En İyi Kategori** | TR Fişleri (%60) |
| **En Sorunlu Kategori** | Makbuzlar (%17) |
| **En Başarılı Alan** | Tarih (%60) |
| **En Sorunlu Alan** | KDV (%36) |

### Tavsiye
1. **Hemen:** TVA ve TOPLAM parser'larını genişletin
2. **Sonra:** Makbuzlar için özel preprocessing ekleyin
3. **Gelecekte:** Google Vision veya Donut ile ensemble deneyin

---

*Rapor otomatik olarak benchmark.py tarafından üretilmiştir.*
