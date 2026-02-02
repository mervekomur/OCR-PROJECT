# FLO OCR Project - Handover Document

## Project Overview

FLO Grup Masraf Modülü için geliştirilen OCR (Optik Karakter Tanıma) sistemi. Fiş ve fatura görüntülerinden yapılandırılmış veri çıkarımı yapar.

## Architecture

### Core Components

```
src/
├── engines/
│   ├── hybrid_vision_engine.py   # Ana OCR motoru (Google Vision + Claude)
│   ├── google_vision_engine.py   # Google Vision API wrapper
│   └── claude_vision_engine.py   # Claude API wrapper
├── logic/
│   └── tax_calculator.py         # KDV hesaplama ve semantic completion
└── models/
    └── receipt.py                # Veri modelleri
```

### Data Flow

```
Image → Google Vision API → Raw Text → Claude (JSON Format) → KDV Heuristics → SAP JSON
         (Source of Truth)           (Formatter Only)        (Mathematical)
```

## Key Business Rules

### KDV (VAT) Calculation

| Durum | KDV Oranı |
|-------|-----------|
| TESK fişi | %0 |
| Yurt dışı masrafı | %0 |
| Taksi (TESK yok) | %20 |
| Tren bileti | %10 |
| Yemek/Konaklama | %10 |
| Genel hizmet | %20 |

### FLO Company VKNs (Alıcı Doğrulama)

- 3880239429 → FLO Mağazacılık
- 8721503797 → Turuncu Ayakkabı
- 3881618492 → FLO Teknoloji
- 3881765897 → FLO İç Dış Ticaret

## API Dependencies

1. **Google Vision API** - OCR text extraction (Source of Truth)
2. **Anthropic Claude API** - JSON formatting only

## Environment Variables

```bash
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Repository Hygiene

### Clean Codebase Policy

Bu proje **temiz versiyon kontrol yapısına** (clean codebase) sahiptir:

#### Repo'dan İzole Edilen Dosyalar

| Kategori | Dosya Türü | Durum |
|----------|------------|-------|
| AI Konuşma Geçmişi | `.claude/`, `*_conversation.json` | `.gitignore`'da |
| Test Sonuçları | `*_sonuclari.json`, `*_results.json` | `.gitignore`'da |
| Hassas Veriler | `*-credentials.json`, `.env` | `.gitignore`'da |
| Geçici Dosyalar | `_temp_*`, `*.log` | `.gitignore`'da |
| Medya Dosyaları | `*.jpg`, `*.png` | `.gitignore`'da |

#### Kurumsal Standartlar

1. **Kod ve Veri Ayrımı**: Test sonuçları ve benchmark çıktıları repo dışında tutulur
2. **Hassas Veri Koruması**: API anahtarları ve credentials asla commit edilmez
3. **Konuşma İzolasyonu**: AI assistant logları ve konuşma geçmişleri repo'ya dahil edilmez
4. **Temiz Commit Geçmişi**: Sadece fonksiyonel kod ve dokümantasyon commit edilir

#### Gitignore Kapsamı

```
.claude/                    # AI assistant dosyaları
*_sonuclari.json           # Test sonuçları
*_results.json             # Benchmark sonuçları
*-credentials.json         # API credentials
.env                       # Environment variables
*.log                      # Log dosyaları
```

### Yeni Geliştirici Rehberi

1. Repo'yu clone edin
2. `.env` dosyası oluşturun (örnek: `.env.example`)
3. Google Cloud credentials dosyasını `credentials/` klasörüne koyun
4. `pip install -r requirements.txt` çalıştırın
5. Test: `python test_hybrid_engine.py`

---

## TODO / Future Work

- [ ] SAP/HR entegrasyonu ile çalışan eşleştirme
- [ ] Batch processing optimizasyonu
- [ ] Web arayüzü (Flask/FastAPI)

---

*Son güncelleme: 2026-02-02*
*Maintainer: FLO Development Team*
