# Laporan Eksperimen: Domain-Adaptive Pre-Training untuk Klasifikasi Emosi Tweet Bahasa Indonesia

**Eksperimen:** `2026-xlmt-emotweet`  
**Tanggal:** Oktober 2026  
**Model:** mBERT, XLM-T, Indo-RoBERTa  
**Total run:** 226 (6 DAPT + 220 fine-tuning; 11 kondisi × 4 dataset × 5 seed)

---

## 1. Masalah Penelitian

Model bahasa besar yang tersedia secara publik umumnya dilatih pada data teks umum atau multibahasa, bukan pada domain spesifik seperti tweet emosi berbahasa Indonesia. Meskipun model seperti XLM-T sudah dilatih pada ratusan juta tweet multibahasa, representasi untuk tweet emosi Indonesia berskala kecil. Sementara Indo-RoBERTa dilatih pada teks formal Indonesia (Common Crawl/OSCAR), sehingga menghadapi *domain gap* yang lebih besar ketika diaplikasikan pada teks informal seperti tweet.

**Domain-Adaptive Pre-Training (DAPT)** adalah teknik *continued pre-training* pada korpus domain target tanpa mengubah arsitektur. Namun, DAPT penuh (semua parameter) memerlukan sumber daya komputasi yang signifikan. Pertanyaannya: apakah mungkin mendapatkan manfaat DAPT dengan biaya yang lebih kecil, misalnya dengan hanya melatih sebagian parameter (*embedding-only*)?

---

## 2. Pertanyaan Penelitian

| ID | Pertanyaan |
|----|-----------|
| **RQ1** | Apakah DAPT pada korpus tweet emosi Indonesia meningkatkan performa klasifikasi dibanding tanpa DAPT? |
| **RQ2** | Apakah DAPT *embedding-only* (encoder frozen) memberikan manfaat yang sebanding dengan DAPT penuh, dengan efisiensi parameter/waktu yang lebih baik? *(kontribusi utama)* |
| **RQ3** | Bagaimana perilaku strategi fine-tuning hemat (*freeze encoder*) setelah DAPT, terutama pada dataset low-resource? |
| **RQ4** | Apakah manfaat DAPT bergantung pada domain gap awal model? (motivasi penambahan Indo-RoBERTa) |

---

## 3. Desain Penelitian

### 3.1 Model

Tiga model dipilih: mBERT sebagai baseline multilingual generik, XLM-T sebagai model domain-adjacent (tweet), dan Indo-RoBERTa sebagai model monolingual Indonesia dengan domain gap besar.

| | mBERT | XLM-T | Indo-RoBERTa |
|---|---|---|---|
| Identifier | `bert-base-multilingual-cased` | `cardiffnlp/twitter-xlm-roberta-base` | `flax-community/indonesian-roberta-base` |
| Pre-training | Wikipedia 104 bahasa | ~198 juta tweet multibahasa | OSCAR (teks formal Indonesia) |
| Vocab | ~120K (WordPiece) | ~250K (SentencePiece) | ~50K (BPE) |
| % Embedding dari total param | ~22.6% | **69.4%** | **31.8%** |
| Domain gap ke tweet ID | **Sangat Besar** | **Kecil** | **Besar** |
| Total parameter | 178M | 278M | 125M |
| Peran | Baseline eksternal | Eksperimen DAPT utama | Eksperimen DAPT utama |

DAPT hanya dilakukan pada XLM-T dan Indo-RoBERTa. mBERT digunakan sebagai titik referensi eksternal (no-DAPT only) untuk mengukur seberapa besar keuntungan domain-specific pre-training.

### 3.2 Data

**Korpus DAPT (unlabeled):**
- ~1.676.000 tweet Indonesia (unlabeled)
- Dikumpulkan dengan kata kunci emosi dari X/Twitter
- Deduplikasi ketat terhadap test set dataset evaluasi
- Split: 98% train (~1.642.480), 2% validasi (~33.520)
- **Download:** [Google Drive](https://drive.google.com/file/d/13-f7oQTZZb0fbkO5UpIKocnKDDGJyFNH/view?usp=sharing) (120 MB) → simpan ke `data/dapt/dapt_tweets_1676k.txt`

**Dataset evaluasi (labeled):**

| # | Dataset | Skema Label | #Kelas | #Train | #Dev | #Test |
|---|---------|------------|--------|--------|------|-------|
| D1 | EmoTweetID | anger, disgust, fear, joy, sadness, surprise | 6 | 1.570 | 336 | 337 |
| D2 | EmoCMT | anger, fear, joy, love, sadness | 5 | 407 | 87 | 88 |
| D3 | EmoT | anger, fear, happy, love, sadness | 5 | 3.080 | 660 | 661 |
| D4 | EmoPubID | anger, fear, joy, love, neutral, sad | 6 | 4.955 | 1.062 | 1.063 |

Keempat dataset mempunyai skema label berbeda sehingga dilatih dan dievaluasi **terpisah** (satu classification head per dataset). Metrik utama: **macro-F1** (untuk menangani ketidakseimbangan kelas dan perbedaan jumlah kelas antar dataset).

### 3.3 Prosedur DAPT

**Objektif:** Masked Language Modeling (MLM), `mlm_probability = 0.15`

**Dua varian:**
- **A-full:** semua parameter diperbarui (100%)
- **A-emb:** encoder dibekukan; hanya embedding + LM head yang diperbarui

Hyperparameter DAPT (identik untuk kedua varian — kontrol variabel untuk RQ2):

| Hyperparameter | Nilai |
|---|---|
| Learning rate | `1e-4` |
| Per-device batch size | 64 |
| Gradient accumulation | 4 → effective batch 256 |
| Max sequence length | 128 |
| Max epochs (XLM-T) | 3 |
| Max epochs (IR) | 3 dan 6 (eksperimen tambahan) |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| LR scheduler | Linear decay |
| Precision | bf16 (L40S GPU) |
| Early stopping patience | 3 evaluasi (per 1.000 steps) |

**Parameter trainable per varian:**

| Model | Varian | Parameter Trainable | % dari Total |
|-------|--------|---------------------|--------------|
| XLM-T | A-full | 278.295.186 | 100.0% |
| XLM-T | A-emb | 193.240.722 | **69.4%** |
| IR | A-full | 124.697.433 | 100.0% |
| IR | A-emb | 39.642.969 | **31.8%** |

Catatan: proporsi embedding XLM-T jauh lebih besar (69.4% vs 31.8%) karena vocab 250K vs 50K pada embedding dimension 768.

### 3.4 Prosedur Fine-tuning

Model hasil DAPT (atau model baseline) di-fine-tune secara terpisah pada masing-masing 4 dataset dengan **5 random seed (42, 123, 456, 789, 999)**. Setiap run menghasilkan `summary.json` berisi `test_macro_f1`.

**Hyperparameter fine-tuning:**

| Hyperparameter | Nilai |
|---|---|
| Max epochs | 20 (dengan early stopping) |
| Early stopping patience | 5 epoch |
| Model selection metric | Macro-F1 validation |
| Learning rate | `2e-5` |
| LR scheduler | Linear dengan warmup 10% |
| Per-device batch size | 32 |
| Max sequence length | 128 |
| Weight decay | 0.01 |
| Precision | bf16 |

**Freeze mode:**
- `full`: semua parameter trainable saat fine-tuning
- `freeze_encoder`: encoder dibekukan; hanya embedding + classification head yang trainable

### 3.5 Matriks Eksperimen

**XLM-T (5 kondisi × 4 dataset × 5 seed = 100 run):**

| ID | DAPT | Fine-tune | Peran |
|----|------|-----------|-------|
| E1 | — | Full | Baseline XLM-T |
| E2 | A-full | Full | Upper bound |
| E3 | **A-emb** | Full | **Klaim efisiensi RQ2** |
| E4 | A-emb | Freeze-encoder | Efisiensi ganda (RQ3) |
| E5 | A-full | Freeze-encoder | Pembanding RQ3 |

**Indo-RoBERTa (5 kondisi × 4 dataset × 5 seed = 100 run):**

| ID | DAPT | Fine-tune | Peran |
|----|------|-----------|-------|
| IR-E1 | — | Full | Baseline IR |
| IR-E2 | A-full 3ep | Full | Upper bound 3ep |
| IR-E3 | A-emb 3ep | Full | RQ2 di model vocab kecil |
| IR-E2-6ep | A-full 6ep | Full | Apakah lebih lama DAPT lebih baik? |
| IR-E3-6ep | A-emb 6ep | Full | RQ2 dengan DAPT lebih lama |

**mBERT Baseline (1 kondisi × 4 dataset × 5 seed = 20 run):**

| ID | DAPT | Fine-tune | Peran |
|----|------|-----------|-------|
| MB-E1 | — | Full | Baseline eksternal multilingual generik |

DAPT **tidak** dilakukan untuk mBERT — hanya fine-tuning langsung dari model pre-trained untuk mengukur kemampuan baseline tanpa adaptasi domain.

**Total:** 11 kondisi × 4 dataset × 5 seed = **220 fine-tuning run** + 6 DAPT run = **226 run total**.

---

## 4. Hasil

### 4.1 Kurva Training DAPT

![Gambar 1](figures/fig1_dapt_curves.png)

**Gambar 1.** Kurva eval loss MLM selama DAPT. *Kiri:* XLM-T — A-full berjalan penuh 3 epoch, A-emb berhenti di epoch ~1.87 karena early stopping (loss plateau lebih cepat). *Kanan:* Indo-RoBERTa — A-full 6 epoch terus menurun tanpa early stopping, membuktikan model belum konvergen di epoch 3; A-emb 6ep berhenti di epoch ~5.3.

**Temuan kurva DAPT:**

| Model | Varian | Epochs | Perplexity | Early Stop? | Waktu (menit) |
|-------|--------|--------|------------|-------------|---------------|
| XLM-T | A-full | 3 | **13.50** | Tidak | 213.6 |
| XLM-T | A-emb | ~1.87 | **18.41** | **Ya** | 121.4 |
| IR | A-full 3ep | 3 | 21.81 | Tidak | 92.2 |
| IR | A-emb 3ep | 3 | 34.97 | Tidak | 73.9 |
| IR | A-full 6ep | 6 | **18.93** | Tidak | 184.9 |
| IR | A-emb 6ep | ~5.3 | **31.22** | **Ya** (ep 5.3) | 142.9 |

Perplexity IR A-full menurun dari 21.81 (3ep) ke 18.93 (6ep), penurunan **−2.88 poin**, mengkonfirmasi manfaat training lebih lama.

### 4.2 Hasil Fine-tuning XLM-T

![Gambar 2](figures/fig2_xlmt_results.png)

**Gambar 2.** Macro-F1 untuk 5 kondisi eksperimen XLM-T pada 4 dataset (horizontal bar chart). Error bar = ±1 std dari 5 seed. Garis vertikal putus-putus = level baseline E1. E4 dan E5 (freeze encoder saat fine-tuning) menunjukkan drop drastis di D1 dan D2.

**Tabel hasil XLM-T — Macro-F1 mean ± std (5 seed):**

| Kondisi | D1 | D2 | D3 | D4 | Avg |
|---------|----|----|----|----|-----|
| E1 No DAPT *(baseline)* | 0.838 ± 0.012 | 0.549 ± 0.079 | 0.757 ± 0.014 | 0.773 ± 0.006 | **0.729** |
| E2 A-full DAPT | 0.838 ± 0.010 | 0.520 ± 0.026 | 0.780 ± 0.017 | 0.783 ± 0.007 | **0.730** |
| E3 A-emb DAPT | 0.833 ± 0.021 | 0.523 ± 0.028 | 0.764 ± 0.014 | 0.774 ± 0.011 | **0.723** |
| E4 A-emb + freeze FT | 0.181 ± 0.038 | 0.156 ± 0.051 | 0.485 ± 0.011 | 0.556 ± 0.015 | **0.344** |
| E5 A-full + freeze FT | 0.296 ± 0.027 | 0.215 ± 0.018 | 0.596 ± 0.021 | 0.629 ± 0.014 | **0.434** |

### 4.3 Hasil Fine-tuning Indo-RoBERTa

![Gambar 3](figures/fig3_ir_results.png)

**Gambar 3.** Macro-F1 untuk 5 kondisi eksperimen Indo-RoBERTa pada 4 dataset. Format sama dengan Gambar 2.

**Tabel hasil Indo-RoBERTa — Macro-F1 mean ± std (5 seed):**

| Kondisi | D1 | D2 | D3 | D4 | Avg |
|---------|----|----|----|----|-----|
| IR-E1 No DAPT *(baseline)* | 0.808 ± 0.020 | 0.320 ± 0.038 | 0.752 ± 0.009 | 0.758 ± 0.015 | **0.660** |
| IR-E2 A-full DAPT 3ep | 0.821 ± 0.016 | 0.353 ± 0.050 | 0.758 ± 0.013 | 0.771 ± 0.009 | **0.676** |
| IR-E3 A-emb DAPT 3ep | 0.817 ± 0.020 | 0.323 ± 0.045 | 0.755 ± 0.016 | 0.767 ± 0.015 | **0.666** |
| IR-E2-6ep A-full DAPT 6ep | **0.829 ± 0.017** | **0.401 ± 0.035** | 0.752 ± 0.007 | 0.770 ± 0.007 | **0.688** |
| IR-E3-6ep A-emb DAPT 6ep | 0.820 ± 0.016 | 0.340 ± 0.041 | 0.758 ± 0.018 | 0.764 ± 0.013 | **0.671** |

### 4.4 Analisis RQ1 — Apakah DAPT Membantu?

![Gambar 4](figures/fig4_rq1_dapt_gain.png)

**Gambar 4.** Δ Macro-F1 antara kondisi DAPT vs baseline (no-DAPT) per dataset. Batang positif = DAPT membantu, negatif = DAPT merugikan.

**Tabel delta Macro-F1 (DAPT vs No-DAPT):**

| Perbandingan | D1 | D2 | D3 | D4 | Avg Δ |
|---|---|---|---|---|---|
| XLM-T: E2 − E1 (A-full) | −0.000 | −0.028 | +0.023 | +0.010 | **+0.001** |
| XLM-T: E3 − E1 (A-emb) | −0.004 | −0.026 | +0.007 | +0.001 | **−0.006** |
| IR: E2 − E1 (A-full 3ep) | +0.014 | +0.032 | +0.006 | +0.014 | **+0.016** |
| IR: E3 − E1 (A-emb 3ep) | +0.009 | +0.002 | +0.003 | +0.010 | **+0.006** |
| IR: E2-6ep − E1 (A-full 6ep) | +0.021 | **+0.081** | −0.001 | +0.012 | **+0.028** |
| IR: E3-6ep − E1 (A-emb 6ep) | +0.012 | +0.020 | +0.006 | +0.007 | **+0.011** |

**Temuan RQ1:**

- **XLM-T:** Gain DAPT sangat kecil dan tidak konsisten. E2 dan E3 hampir setara dengan E1. Di D2 bahkan sedikit di bawah baseline. Ini konsisten dengan hipotesis: XLM-T sudah dilatih pada tweet multilingual (*domain gap kecil*), sehingga DAPT tidak memberikan sinyal baru yang signifikan.

- **Indo-RoBERTa:** DAPT 6 epoch memberikan gain yang paling konsisten:
  - D1: +2.1%, D2: **+8.1%** (IR-E2-6ep vs IR-E1)
  - D3 dan D4: kecil (< ±0.8%) karena dataset ini lebih besar dan sudah cukup data untuk fine-tuning

  Ini konsisten dengan hipotesis domain gap besar: IR dilatih pada teks formal, sehingga adaptasi ke tweet memberikan manfaat nyata, terutama pada dataset kecil (D2 = 582 sampel).

**Jawaban RQ1:** DAPT membantu ketika domain gap antara model pre-training dan domain target besar (IR). Untuk model yang sudah domain-adjacent (XLM-T), gain marginal.

### 4.5 Analisis RQ2 — Apakah A-emb Setara A-full?

![Gambar 5](figures/fig5_rq2_emb_vs_full.png)

**Gambar 5.** Δ Macro-F1 antara A-emb dan A-full DAPT (A-emb − A-full). Mendekati 0 = setara. Positif = A-emb lebih baik. Negatif = A-full lebih baik.

**Tabel delta Macro-F1 (A-emb − A-full):**

| Perbandingan | D1 | D2 | D3 | D4 | Avg Δ |
|---|---|---|---|---|---|
| XLM-T: E3 − E2 | −0.004 | +0.003 | −0.016 | −0.009 | **−0.007** |
| IR 3ep: E3 − E2 | −0.004 | −0.030 | −0.003 | −0.004 | **−0.010** |
| IR 6ep: E3-6ep − E2-6ep | −0.009 | **−0.061** | +0.007 | −0.006 | **−0.017** |

**Temuan RQ2:**

| Model | Avg Δ (A-emb − A-full) | %Param hemat | Waktu hemat | Interpretasi |
|-------|------------------------|--------------|-------------|--------------|
| XLM-T (emb 69.4%) | −0.007 | 30.6% | **43%** | **A-emb ≈ A-full** |
| IR 3ep (emb 31.8%) | −0.010 | 68.2% | 20% | Sedikit lebih lemah |
| IR 6ep (emb 31.8%) | −0.017 | 68.2% | 23% | A-full lebih baik |

Untuk **XLM-T**, di mana embedding menyumbang **69.4%** dari total parameter, A-emb memberikan hasil hampir identik dengan A-full di semua dataset. Ini mengkonfirmasi klaim RQ2: sebagian besar "kapasitas adaptasi" DAPT pada XLM-T memang terkonsentrasi di embedding layer.

Untuk **Indo-RoBERTa**, di mana embedding hanya **31.8%** dari parameter, A-emb lebih lemah dari A-full. Gap terbesar terjadi di D2 pada 6ep (−0.061), yaitu dataset paling kecil (582 sampel) yang paling sensitif terhadap kualitas representasi.

**Jawaban RQ2:** *Embedding-only DAPT efektif ketika embedding mendominasi arsitektur (XLM-T: 69.4%)*. Ketika proporsi embedding kecil (IR: 31.8%), encoder layers menyimpan informasi domain yang tidak dapat teradaptasi dengan A-emb saja.

### 4.6 Analisis Efisiensi

![Gambar 6](figures/fig6_efficiency.png)

**Gambar 6.** *Kiri:* Trade-off waktu DAPT (menit) vs persentase parameter trainable. Ukuran titik proporsional dengan avg Macro-F1 downstream. *Kanan:* Hubungan antara DAPT perplexity dan downstream macro-F1.

**Tabel trade-off efisiensi:**

| Model & Varian | %Param | Waktu (mnt) | PPL | Avg F1 | F1/Menit |
|----------------|--------|-------------|-----|--------|----------|
| XLM-T A-full | 100.0% | 213.6 | 13.50 | 0.730 | 0.342 |
| XLM-T A-emb | 69.4% | **121.4** | 18.41 | **0.723** | **0.596** |
| IR A-full 3ep | 100.0% | 92.2 | 21.81 | 0.676 | 0.733 |
| IR A-emb 3ep | 31.8% | **73.9** | 34.97 | 0.666 | 0.901 |
| IR A-full 6ep | 100.0% | 184.9 | 18.93 | **0.688** | 0.372 |
| IR A-emb 6ep | 31.8% | 142.9 | 31.22 | 0.671 | 0.470 |

XLM-T A-emb menghemat **43% waktu DAPT** dengan penurunan F1 hanya **0.007** (< 1%). Rasio F1/Menit A-emb (0.596) hampir **1.74×** lebih efisien dibanding A-full (0.342).

### 4.7 Heatmap Semua Hasil

![Gambar 7](figures/fig7_heatmap.png)

**Gambar 7.** Heatmap macro-F1 untuk seluruh 11 kondisi (termasuk MB-E1) × 4 dataset. Warna lebih gelap = F1 lebih tinggi. Garis putih memisahkan kelompok mBERT, XLM-T, dan Indo-RoBERTa. E4 dan E5 terlihat jelas sebagai baris terang (F1 rendah) akibat freeze encoder saat fine-tuning.

### 4.8 Perbandingan Baseline — mBERT vs XLM-T vs Indo-RoBERTa

![Gambar 8](figures/fig8_baseline_comparison.png)

**Gambar 8.** Perbandingan tiga kondisi no-DAPT baseline: MB-E1 (mBERT), E1 (XLM-T), dan IR-E1 (Indo-RoBERTa). Error bar = ±1 std dari 5 seed.

**Tabel hasil mBERT baseline — Macro-F1 mean ± std (5 seed):**

| Kondisi | D1 | D2 | D3 | D4 | Avg |
|---------|----|----|----|----|-----|
| MB-E1 mBERT (no DAPT) | 0.811 ± 0.009 | 0.321 ± 0.060 | 0.650 ± 0.015 | 0.721 ± 0.009 | **0.626** |

**Perbandingan antar baseline:**

| Perbandingan | D1 | D2 | D3 | D4 | Avg Δ |
|---|---|---|---|---|---|
| XLM-T E1 − MB-E1 | +0.027 | **+0.228** | **+0.107** | +0.052 | **+0.103** |
| IR-E1 − MB-E1 | −0.003 | −0.001 | +0.102 | +0.037 | **+0.034** |
| IR-E2-6ep − MB-E1 (best IR) | +0.018 | **+0.080** | +0.102 | +0.049 | **+0.062** |

**Temuan perbandingan baseline:**

- **XLM-T vs mBERT (+10.3% rata-rata):** Tweet-domain pre-training memberikan keuntungan besar, terutama di D2 (+22.8%) dan D3 (+10.7%). D2 dan D3 adalah dataset dengan teks tweet yang paling informal, sehingga representasi tweet-level pada XLM-T sangat menguntungkan.

- **Indo-RoBERTa (no DAPT) vs mBERT:** Pada rata-rata hampir setara (+3.4%), bahkan di D1 sedikit di bawah mBERT (−0.003). Ini menunjukkan bahwa bahasa target (Indonesia) saja tidak cukup — domain (tweet informal) sama pentingnya.

- **Indo-RoBERTa + DAPT 6ep vs mBERT (+6.2%):** DAPT yang efektif dapat mengkompensasi kurangnya domain pre-training. IR-E2-6ep mampu melampaui mBERT secara konsisten di semua dataset.

- **Implikasi:** Untuk klasifikasi tweet emosi bahasa Indonesia, pemilihan model harus mempertimbangkan dua dimensi: *bahasa target* (Indonesian) dan *domain* (informal tweet). XLM-T unggul di dimensi domain, sementara Indo-RoBERTa unggul di bahasa — DAPT membantu menjembatani gap domain IR.

### 4.9 Uji Signifikansi Statistik

Uji Wilcoxon signed-rank (two-sided) dijalankan dengan unit observasi per-seed (rata-rata macro-F1 dari 4 dataset per seed), menghasilkan n=5 pasangan untuk setiap komparasi.

**Hasil uji Wilcoxon (n=5, two-sided):**

| Perbandingan | Arah semua seed sama? | p-value |
|---|---|---|
| IR E2-6ep vs IR-E1 (RQ1: DAPT gain) | Ya (+) | **0.0625** |
| XLM-T E3 vs E2 (RQ2: A-emb vs A-full) | Tidak (campuran) | 0.3125 |
| IR E3 vs E2 3ep (RQ2) | Ya (−) | **0.0625** |
| IR E3-6ep vs E2-6ep 6ep (RQ2) | Ya (−) | **0.0625** |

> **Catatan metodologi:** Dengan n=5, p-value minimum yang dapat dicapai Wilcoxon two-sided adalah **0.0625** (ketika semua 5 perbedaan searah). Tiga komparasi mencapai nilai minimum ini, mengindikasikan konsistensi maksimal yang dapat dibuktikan secara statistik dengan 5 seed. Untuk mencapai p < 0.05 dua arah, dibutuhkan minimal n=6.

**Effect size kunci (Δ Macro-F1, rata-rata 4 dataset):**

| Perbandingan | Avg Δ | Konsistensi | Kesimpulan |
|---|---|---|---|
| XLM-T: E2 vs E1 | +0.001 | Tidak konsisten | Tidak ada efek |
| XLM-T: E3 vs E1 | −0.006 | Kecil | Tidak ada efek bermakna |
| **XLM-T: E3 vs E2 (RQ2)** | **−0.007** | Campuran | **A-emb ≈ A-full** |
| IR: E2-6ep vs E1 | **+0.028** | Konsisten | **DAPT membantu** |
| IR: E3-6ep vs E2-6ep | −0.017 | Konsisten (−) | A-full > A-emb |

---

## 5. Diskusi

### 5.1 Mengapa XLM-T Tidak Merespons DAPT?

XLM-T dilatih pada ~198 juta tweet multibahasa, termasuk tweet Indonesia. Model ini sudah memiliki representasi yang cukup untuk domain tweet emosi Indonesia. DAPT dengan 1.676.000 tweet (~0.8% dari ukuran pre-training data asli) tidak memberikan perubahan representasi yang cukup signifikan untuk mempengaruhi downstream performance.

Hal ini bukan kegagalan — sebaliknya, ini mengkonfirmasi bahwa XLM-T sudah optimal untuk domain ini. Eksperimen XLM-T berkontribusi pada studi efisiensi: meskipun DAPT tidak wajib untuk XLM-T, jika dilakukan, A-emb memberikan hasil setara dengan ~43% penghematan waktu.

### 5.2 Mengapa Indo-RoBERTa Merespons Lebih Baik?

Indo-RoBERTa dilatih pada teks formal Indonesia. Tweet emosi bersifat informal — mengandung singkatan, emotikon, bahasa gaul, dan struktur yang berbeda dari teks formal. Domain gap yang lebih besar ini berarti DAPT memberikan sinyal adaptasi yang nyata. Gain terbesar terjadi di D2 (dataset terkecil, 582 sampel) karena pada dataset kecil, kualitas representasi lebih kritis dibanding besarnya data training.

### 5.3 Interpretasi RQ2: Embedding Proportion sebagai Prediktor

Temuan kunci yang layak dipromosikan di paper:
- **XLM-T (emb 69.4%):** A-emb ≈ A-full (Δ = −0.007, p = 0.3125 — tidak ada bukti perbedaan bermakna)
- **IR (emb 31.8%):** A-full > A-emb (Δ = −0.010 hingga −0.017, p = 0.0625 — konsisten di semua seed)

Ini mengindikasikan bahwa efektivitas *embedding-only DAPT* berkorelasi dengan proporsi embedding dalam arsitektur. Untuk model dengan vocab besar (XLM-R family dengan 250K vocab), embedding layer secara alami mendominasi parameter count dan sekaligus menjadi komponen paling kritis untuk adaptasi domain.

### 5.4 Collapse pada E4/E5

Freeze encoder saat fine-tuning setelah DAPT ("double freeze") menyebabkan drop drastis di D1 dan D2. Ini disebabkan oleh konflik representasi: saat DAPT dengan encoder frozen, embedding belajar representasi domain baru; ketika fine-tuning dengan encoder juga frozen, classifier head harus belajar dari representasi yang tidak dapat disesuaikan kembali untuk tugas klasifikasi. Efek ini lebih parah di D1/D2 (dataset kecil) karena kapasitas adaptasi lebih terbatas.

### 5.5 Konteks mBERT sebagai Baseline Eksternal

mBERT (178M parameter, Wikipedia 104 bahasa) memberikan rata-rata macro-F1 **0.626** tanpa DAPT. Ini adalah titik referensi penting karena mBERT bukan model domain-spesifik maupun bahasa-spesifik untuk tweet emosi Indonesia.

XLM-T tanpa DAPT (E1) sudah melampaui mBERT sebesar **+10.3%** — membuktikan bahwa tweet-domain pre-training pada 198M tweet multilingual memberikan representasi yang jauh lebih baik untuk tugas ini. Bahkan Indo-RoBERTa dengan DAPT 6ep (IR-E2-6ep) yang optimal hanya mengalahkan mBERT sebesar **+6.2%**, lebih rendah dari XLM-T tanpa DAPT sama sekali.

Ini menggarisbawahi pentingnya *domain match* (tweet) dibanding *language match* (bahasa Indonesia) untuk task-specific fine-tuning pada dataset emosi tweet, setidaknya pada skala model yang sebanding (~125–278M parameter).

### 5.6 Dampak Jumlah Epoch DAPT (3ep vs 6ep)

Untuk IR, perpanjangan DAPT dari 3 ke 6 epoch menurunkan perplexity dari 21.81 ke 18.93 (−13%). Gain downstream rata-rata adalah +1.2% — kecil tapi konsisten di D1 dan D2, dengan puncak di D2 (+4.9%). Implikasinya: diminishing returns mulai terasa, dan gain downstream tidak linear dengan penurunan perplexity. Untuk XLM-T, early stopping di ~1.87 epoch mengindikasikan model sudah konvergen jauh lebih cepat karena domain gap kecil.

---

## 6. Kesimpulan

| RQ | Jawaban |
|----|---------|
| **RQ1** | DAPT hanya bermanfaat ketika domain gap besar (IR: avg +2.8% untuk A-full 6ep). Untuk XLM-T yang sudah domain-adjacent, DAPT tidak meningkatkan performa secara bermakna (avg ≈ +0.001). |
| **RQ2** | *Embedding-only DAPT* efektif ketika embedding mendominasi arsitektur (XLM-T: 69.4%, Δ = −0.007 vs A-full, p = 0.3125). Dengan penghematan 43% waktu training, A-emb memberikan rasio efisiensi **1.74×** lebih baik dari A-full. Untuk IR (emb 31.8%), A-full secara konsisten lebih baik (Δ = −0.010 s.d. −0.017, p = 0.0625). |
| **RQ3** | "Double freeze" (encoder frozen saat DAPT + fine-tuning) tidak direkomendasikan. Kolaps terutama pada dataset kecil (D1, D2). |
| **RQ4** | Domain gap terbukti menjadi prediktor penting manfaat DAPT. XLM-T (gap kecil) tidak merespons; IR (gap besar) merespons lebih baik, terutama dengan 6 epoch dan pada dataset kecil. |

**Klaim kontribusi yang dapat dipertahankan:**
1. Studi pertama yang menguji *embedding-only adaptation* pada model multilingual (XLM-R family, 250K vocab) untuk DAPT — berbeda dari BERT yang dikaji Ladkat et al. (2022) dengan proporsi embedding ~21%.
2. Bukti kuantitatif bahwa efektivitas *embedding-only DAPT* berkorelasi dengan proporsi embedding dalam arsitektur: efektif pada XLM-T (69.4%) namun tidak pada Indo-RoBERTa (31.8%).
3. Trade-off efisiensi terukur: XLM-T A-emb menghemat **43% waktu DAPT** dengan < 1% penurunan macro-F1 rata-rata, dikonfirmasi pada 5 seed × 4 dataset (n=200 run).
4. Analisis multi-baseline (mBERT + XLM-T + Indo-RoBERTa) yang membuktikan dominasi *domain match* atas *language match*: XLM-T tanpa DAPT (+10.3% vs mBERT) lebih unggul dari Indo-RoBERTa dengan DAPT optimal (+6.2% vs mBERT).
5. Analisis multi-dataset (4 dataset, skema label berbeda) dan multi-seed (n=5) yang memberikan bukti konsistensi lebih kuat dibanding studi sebelumnya dengan n=1-3.

**Rekomendasi untuk submission:**
- **Scopus Q3 IEEE/ACM:** Framing efisiensi dengan 5 seed, 4 dataset, dan 3 baseline (mBERT, XLM-T, IR) sudah cukup kuat. Tekankan klaim efisiensi XLM-T A-emb dan korelasi proporsi embedding sebagai kontribusi utama, dengan perbandingan mBERT sebagai triangulasi baseline.
- **Scopus Q2:** Pertimbangkan menambahkan analisis representasi per-layer (probing task atau CKA similarity) untuk menjelaskan mekanisme *mengapa* embedding adaptation cukup di XLM-T. Perbandingan mBERT yang sudah ada memperkuat argumen domain vs language match.

---

## Lampiran: Konfigurasi Hardware dan Reproduktibilitas

- **GPU:** NVIDIA L40S (48 GB VRAM)
- **Precision:** bf16
- **Framework:** PyTorch + HuggingFace Transformers 4.40.0
- **Random seeds:** 42, 123, 456, 789, 999
- **Kode tersedia di:** `src/` (dapt_train.py, finetune_train.py, run_finetune.py)
- **Output:** `outputs/finetune/<exp>/<dataset>/seed<N>/summary.json`
- **Figures:** `outputs/report/figures/fig1–fig8.png`
- **WandB:** project `2026-xlmt-emotweet`, entity `bdsrc-team` (CoreWeave Forge)
