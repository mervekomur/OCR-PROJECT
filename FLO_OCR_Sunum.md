---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #1a5f7a;
  }
  h2 {
    color: #2c3e50;
  }
  table {
    font-size: 0.8em;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
---

# FLO OCR Projesi
## Teknik Evrim ve Mühendislik Yolculuğu

**Tarih:** Şubat 2026
**Süre:** 19 Gün (14 Ocak - 2 Şubat 2026)

---

# Gündem

1. **Proje Özeti** - Ne yaptık?
2. **Teknik Evrim** - Hangi yollardan geçtik?
3. **Kırılma Noktaları** - Kritik kararlar
4. **Metrikler** - Nereden nereye?
5. **Final Mimari** - Hibrit sistem
6. **Mühendislik Vizyonu** - Gelecek

---

# Proje Özeti

## FLO Grup Masraf Modülü v4

| Metrik | Değer |
|--------|-------|
| **Toplam Süre** | 19 gün |
| **Python Dosyası** | 98 adet |
| **OCR Motoru Denendi** | 14 adet |
| **Kod Satırı** | 24.243+ |
| **Hesap Kodu** | 66 adet |
| **Final Accuracy** | %94 |

---

# Problem Tanımı

## Çözülmesi Gereken Sorunlar

- Fiş/fatura görüntülerinden **yapılandırılmış veri** çıkarma
- **KDV hesaplama** ve doğrulama
- **FLO şirket validasyonu** (alıcı kontrolü)
- **SAP uyumlu** JSON çıktısı
- **KVKK uyumlu** veri işleme

---

# Teknik Evrim: 4 Faz

```
Faz 1          Faz 2          Faz 3          Faz 4
Ensemble  ──▶  SAM      ──▶  Google   ──▶  Hybrid
(14-15 Oca)    (19 Oca)       Vision        Engine
                              (30 Oca)       (2 Şub)
   │              │              │              │
   ▼              ▼              ▼              ▼
Multi-OCR    Fiş Tespit    Cloud API    Final Mimari
```

---

# Faz 1: Ensemble Mimarisi
## 14-15 Ocak 2026

### Denenen Motorlar
- **PaddleOCR** - Türkçe desteği
- **EasyOCR** - Genel amaçlı
- **Donut** - Document Understanding
- **GOT-OCR** - End-to-end

### Sonuç
❌ Tek motor yeterli değil → Ensemble pattern

---

# Faz 2: SAM Entegrasyonu
## 19 Ocak 2026

### Segment Anything Model

```
Görüntü ──▶ SAM ──▶ Fiş Tespiti ──▶ Crop ──▶ OCR
```

### Özellikler
- Düşük kontrastlı görüntülerde başarılı
- Otomatik maske seçimi
- Model boyutu: ~375MB

---

# Faz 3: Google Vision API
## 30 Ocak 2026

### Neden Google Vision?

| Özellik | Değer |
|---------|-------|
| OCR Doğruluğu | ~%90 |
| Dil Desteği | 4+ dil |
| Güvenilirlik | Enterprise-grade |
| Hız | Cloud-based |

### Kritik Karar
> Google Vision = **Source of Truth**

---

# Faz 4: Hybrid Vision Engine
## 2 Şubat 2026 - Final Mimari

```
┌─────────────────────────────────────────┐
│           HYBRID VISION ENGINE          │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Google Vision │       │   Claude API  │
│ Source of     │       │  Formatter    │
│ Truth (OCR)   │       │  Only         │
└───────┬───────┘       └───────┬───────┘
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  KDV Heuristics       │
        │  Business Rules       │
        └───────────────────────┘
```

---

# Kırılma Noktası #1
## Preprocessing Keşfi

### Ablation Study Sonuçları (28 Ocak)

| Yöntem | Başarı |
|--------|--------|
| **Original (işlenmemiş)** | **%40.7** ✅ |
| Grayscale | %40.7 |
| Otsu Threshold | %37.0 |
| Adaptive Threshold | %14.8 ❌ |

### Sonuç
> **Preprocessing görüntüyü BOZUYOR!**
> Original en iyi sonucu veriyor.

---

# Kırılma Noktası #2
## RAM Optimizasyonu (23 Ocak)

### Uygulanan 3 Kritik Adım

1. **Lazy Loading**
   - Modeller ilk çağrıda yükleniyor
   - Başlangıç süresi: ~0 saniye

2. **Singleton Pattern**
   - Model instance tek sefer oluşturuluyor
   - Bellek tasarrufu: ~60%

3. **Batch Processing**
   - 19 dosya, 4785 satır yeni kod

---

# Kırılma Noktası #3
## KDV Mathematical Heuristics

### Algoritma (FLO Doc 7.1-7.2)

```
Adım 1: OCR'dan KDV var mı? → Kullan
Adım 2: Yoksa Reverse Calculation:
        • %20: matrah = brut / 1.20
        • %10: matrah = brut / 1.10
Adım 3: Cross-Check (%95 tolerans)
        • Hesaplanan tutar metinde var mı?
Adım 4: Semantic Completion
        • Hasarlı kelimeleri tamamla
```

---

# Kırılma Noktası #4
## Source of Truth Kararı

### Problem
Claude Vision tek başına kullanıldığında **veri uydurma** riski

### Çözüm

| Bileşen | Rol |
|---------|-----|
| **Google Vision** | OCR metin çıkarma (Truth) |
| **Claude API** | JSON formatı (Formatter only) |
| **KDV Heuristics** | Matematiksel hesaplama |

> Claude **hiçbir veri üretmez**, sadece formatlar!

---

# Metrikler: Nereden Nereye?

## Accuracy Karşılaştırması

| Metrik | İlk Prototip | Final API | İyileşme |
|--------|-------------|-----------|----------|
| **Genel Accuracy** | %32 | %94 | **+194%** |
| **Tarih Doğruluğu** | %38 | %92+ | **+142%** |
| **Toplam Tutar** | %59 | %100 | **+69%** |
| **KDV Oranı** | %21 | %85+ | **+305%** |

---

# Metrikler: API Performansı

## Şu Anki Kapasite

| Metrik | Değer |
|--------|-------|
| **İşleme Süresi** | ~25 saniye/belge |
| **RPS** | 0.8 belge/saniye |
| **Token/Belge** | ~4166 token |
| **Kapasite** | ~48 belge/dakika |
| **Confidence** | %94 ortalama |

---

# Business Rules Engine

## 66 Hesap Kodu Eşleştirmesi

| Kategori | Hesap Kodu | KDV | KKEG |
|----------|-----------|-----|------|
| Yemek Gideri | 760.01.001 | %10 | - |
| Taksi (TESK) | 760.02.001 | %0 | - |
| Taksi (Normal) | 760.02.001 | %20 | - |
| Akaryakıt | 760.02.004 | %20 | ✅ |
| Araç Kiralama | 760.02.003 | %20 | ✅ |
| Konaklama | 760.04.001 | %10 | - |

---

# Business Rules: FLO Validasyonu

## Alıcı (Buyer) Kontrolü

### Geçerli FLO Şirketleri

| VKN | Şirket Adı |
|-----|------------|
| 3880239429 | FLO Mağazacılık |
| 8721503797 | Turuncu Ayakkabı |
| 3881618492 | FLO Teknoloji |
| 3881765897 | FLO İç Dış Ticaret |

### Red Kuralları
- ❌ TCKN (11 hane) = Şahıs → RED
- ❌ VKN ∉ FLO listesi → RED

---

# Business Rules: TESK Kuralı

## Taksi Fişlerinde KDV

```
if "TESK" in fiş_metni:
    kdv_orani = 0%    # Esnaf muafiyeti
else:
    kdv_orani = 20%   # Normal oran
```

### Örnek
- **Taksi + TESK yazısı** → KDV %0
- **Taksi + TESK yok** → KDV %20
- **Yurt dışı masraf** → KDV %0

---

# Warning Sistemi

## Otomatik Uyarı Kodları

| Kod | Açıklama |
|-----|----------|
| W001 | TCKN tespit edildi (şahıs) |
| W002 | Brüt tutar = 0 |
| W003 | KDV bilgisi eksik |
| W004 | Matematiksel tutarsızlık |
| W005 | Yurt dışı masraf tespit |
| W006 | Belirsiz kategori |
| W007 | Ghost text (OCR hatası) |

---

# Final Mimari: Data Flow

```
┌──────────┐
│  Görüntü │
└────┬─────┘
     ▼
┌──────────────┐
│ Google Vision│ ──▶ Raw OCR Text
└────┬─────────┘
     ▼
┌──────────────┐
│  Claude API  │ ──▶ Structured JSON
└────┬─────────┘
     ▼
┌──────────────┐
│KDV Heuristics│ ──▶ Tax Calculation
└────┬─────────┘
     ▼
┌──────────────┐
│Business Rules│ ──▶ Validation + KKEG
└────┬─────────┘
     ▼
┌──────────────┐
│  SAP JSON    │
└──────────────┘
```

---

# Proje Yapısı

```
src/
├── engines/                 # 14 OCR motoru
│   ├── base.py              # Abstract pattern
│   ├── hybrid_vision_engine.py  # ⭐ Ana motor
│   ├── google_vision_engine.py
│   └── claude_vision_engine.py
├── logic/
│   └── tax_calculator.py    # KDV heuristics
├── parser/
│   └── item_parser.py       # Tablo parsing
└── sam_detector.py          # Fiş tespiti
```

---

# Ezgi Hanım'dan Kritik Geri Bildirimler

## Mimariyi Değiştiren Kararlar

| Geri Bildirim | Etki |
|---------------|------|
| "Alıcı VKN FLO olmalı" | 4 şirket whitelist |
| "KDV 7.1-7.2'ye göre" | Reverse calculation |
| "TESK varsa KDV %0" | TESK regex pattern |
| "11 hane = şahıs = RED" | TCKN warning sistemi |
| "Veri uydurmamalı" | Source of Truth kararı |

---

# KVKK Uyumluluk

## Veri Koruma Önlemleri

### Git Geçmişi Temizliği
```bash
git filter-branch --index-filter
# Tüm görsel dosyalar silindi
# PII izolasyonu sağlandı
```

### .gitignore Kapsamı
- ✅ Görsel dosyalar (*.jpg, *.png)
- ✅ API credentials
- ✅ Test sonuçları
- ✅ AI assistant dosyaları

---

# Mühendislik Dokümantasyonu

## Devir Teslim Dosyaları

| Dosya | İçerik |
|-------|--------|
| `HANDOVER.md` | Mimari, data flow, TODO |
| `README.md` | Kurulum rehberi |
| `PROJE_OZETI.md` | High-level özet |

### Modüler Yapı Avantajları
- Plug & play motor sistemi
- Lazy loading pattern
- Ensemble fallback

---

# Gelecek: TODO Listesi

## Planlanan İyileştirmeler

- [ ] **SAP/HR Entegrasyonu**
  - Çalışan eşleştirme (Levenshtein)

- [ ] **Batch Optimization**
  - Paralel işleme

- [ ] **Web Arayüzü**
  - Flask/FastAPI dashboard

- [ ] **RPS İyileştirme**
  - Caching layer

---

# Öğrenilen Dersler

## Teknik Insights

1. **Preprocessing her zaman iyi değil**
   - Original görüntü en iyi sonucu verebilir

2. **Hibrit yaklaşım güçlü**
   - Tek model yetersiz kalabilir

3. **Source of Truth kritik**
   - AI'ın veri üretmesine izin verme

4. **Business rules kod içinde**
   - 66 hesap kodu = domain knowledge

---

# Özet: Sayılarla Proje

<div class="columns">
<div>

### Girdiler
- 🕐 19 gün
- 📁 98 Python dosyası
- 🔧 14 OCR motoru
- 📝 24.243 satır kod

</div>
<div>

### Çıktılar
- ✅ %94 accuracy
- 📊 66 hesap kodu
- 🏢 4 FLO şirketi
- 🔒 KVKK uyumlu

</div>
</div>

---

# Teşekkürler

## Sorular?

---

**Proje:** FLO OCR Masraf Modülü v4
**Mimari:** Hybrid Vision Engine
**Teknolojiler:** Google Vision + Claude + Python

---

# Ek: Teknik Detaylar

---

# Ek A: OCR Motor Karşılaştırması

| Motor | Accuracy | Hız | Kullanım |
|-------|----------|-----|----------|
| Google Vision | %90+ | Hızlı | ⭐ Production |
| Claude Vision | %85+ | Orta | Formatting |
| PaddleOCR | %40 | Hızlı | Deprecated |
| EasyOCR | %35 | Orta | Deprecated |
| Donut | %30 | Yavaş | Experimental |

---

# Ek B: KDV Oranları Tablosu

| Senaryo | KDV | Kural |
|---------|-----|-------|
| TESK fişi | %0 | "TESK" kelimesi var |
| Yurt dışı | %0 | Yabancı dil tespit |
| Yemek/Konaklama | %10 | Kategori bazlı |
| Tren bileti | %10 | TCDD/E-bilet |
| Taksi (TESK yok) | %20 | Default taksi |
| Genel hizmet | %20 | Default oran |

---

# Ek C: KKEG Hesaplama

## Binek Araç Giderleri Formülü

```
Matrah      = Brüt / 1.20
Gider       = Brüt × 0.70
KKEG Gider  = Matrah × 0.30
KKEG KDV    = KKEG Gider × 0.20
```

### Örnek: 1000 TL Araç Kiralama
- Matrah: 833.33 TL
- Gider: 700 TL
- KKEG Gider: 250 TL
- KKEG KDV: 50 TL

---

# Ek D: API Kullanımı

```python
from src.engines.hybrid_vision_engine import HybridVisionEngine

# Engine başlat
engine = HybridVisionEngine()

# Görüntü işle
result = engine.extract("fis.jpg")

# Sonuçları al
print(result.fields)
# {
#   "header": {...},
#   "financials": {...},
#   "items": [...],
#   "validation": {...}
# }
```
