# Project YOLO: Deteksi Fork, Spoon, dan Cup

Proyek ini merupakan pipeline pendeteksian objek berbasis YOLO yang dirancang untuk mengolah data dari COCO dan memfokuskan pada tiga kelas utama, yaitu:

- fork
- spoon
- cup

Tujuan dari proyek ini adalah membangun model deteksi objek yang mampu mengenali benda-benda tersebut secara otomatis pada citra yang diambil dari dataset COCO. Proyek ini dibuat dalam format eksperimen pembelajaran mesin yang mencakup tahap pengumpulan data, pengecekan anotasi, pembagian dataset, pelatihan model, hingga evaluasi hasil.

## 1. Gambaran Umum Proyek

Project ini terdiri dari beberapa skrip Python yang berurutan, yaitu:

1. 00_ambil_subset_coco.py  
   Membuat subset dataset dari COCO untuk kelas yang dipilih, seperti fork, spoon, dan cup.

2. 01_cek_anotasi.py  
   Mengecek kualitas anotasi format YOLO agar label yang dihasilkan valid untuk training.

3. 02_yolo_ke_coco.py  
   Mengubah format hasil YOLO kembali ke format COCO (jika diperlukan).

4. 03_coco_ke_yolo.py  
   Mengubah data COCO menjadi format YOLO agar sesuai dengan kebutuhan pelatihan model.

5. 05_split_dataset_yolo.py  
   Membagi dataset menjadi data train, validasi, dan test.

6. 06_latih_bandingkan.py  
   Melatih model YOLO dalam beberapa eksperimen, membandingkan hasil, dan memilih model terbaik.

Secara umum, alur kerja proyek adalah:

- memilih kelas dari COCO,
- mengekstrak subset gambar dan anotasi,
- membangun dataset YOLO,
- memeriksa kualitas label,
- membagi dataset,
- melatih beberapa model untuk membandingkan performa,
- mengevaluasi hasil terbaik pada data validasi dan test.

## 2. Kegunaan Project

Project ini memiliki manfaat penting, yaitu:

- menyediakan dataset deteksi objek kecil yang relevan untuk benda rumah tangga,
- mempermudah proses eksperimen deteksi objek tanpa perlu dataset yang terlalu besar,
- menjadi dasar untuk analisis perbandingan model, seperti model pretrained vs scratch vs tuning,
- memfasilitasi evaluasi performa model melalui metrik seperti mAP50, mAP50-95, precision, dan recall.

## 3. Dataset yang Digunakan

Dataset yang dipakai berasal dari COCO (Common Objects in Context). Namun, tidak semua kelas dalam COCO digunakan. Proyek ini hanya memilih kelas tertentu yang relevan dengan kasus penelitian, yaitu:

- fork
- spoon
- cup

Dengan pendekatan subset, dataset yang digunakan lebih kecil, lebih ringkas, dan lebih fokus untuk kebutuhan eksperimen.

## 4. Struktur Dataset Hasil

Setelah proses pengambilan data, dataset akan terbentuk dengan struktur seperti berikut:

```text
dataset_yolo/
├── classes.txt
├── images/
│   ├── ...jpg
│   └── ...png
├── labels/
│   ├── ...txt
│   └── ...txt
└── cek_visual/
    └── ...gambar dengan bounding box
```

File `classes.txt` berisi urutan kelas yang digunakan, sedangkan file label berisi koordinat bounding box dalam format YOLO.

## 5. Proses Validasi Anotasi

Sebelum pelatihan, anotasi harus dicek dengan cermat. Hal ini dilakukan dengan skrip 01_cek_anotasi.py. Tujuan pengecekan ini adalah memastikan:

- setiap gambar memiliki label yang sesuai,
- format label sesuai dengan format YOLO,
- class_id valid sesuai dengan kelas yang didefinisikan,
- nilai koordinat berada dalam rentang normalisasi 0 sampai 1,
- bounding box tidak terlalu kecil,
- tidak ada label yang rusak atau hilang.

Jika ada masalah pada anotasi, hasil model dapat menurun drastis karena data training tidak konsisten.

## 6. Pelatihan Model

Pada tahap training, skrip 06_latih_bandingkan.py melakukan tiga eksperimen penting:

- E1: pretrained model
- E2: scratch model
- E3: tuning model dengan satu parameter yang diubah

Tujuan eksperimen ini adalah membandingkan model yang sudah terlatih sebelumnya dan model yang dilatih dari awal, serta melihat dampak perubahan arsitektur atau hyperparameter terhadap hasil.

## 7. Alasan Mengubah Epoch dari 30 Menjadi 21 pada Proses Training

Pada awal perencanaan, jumlah epoch yang ditentukan adalah 100 epoch. Namun dalam praktik pelatihan, proses sering berhenti lebih cepat karena diterapkan mekanisme early stopping.

Early stopping adalah teknik untuk menghentikan training secara otomatis ketika model sudah tidak menunjukkan perbaikan yang signifikan dalam jumlah iterasi tertentu. Biasanya, model akan dihentikan ketika metric validasi tidak lagi meningkat selama beberapa epoch terakhir.

### Mengapa epoch bisa berhenti di 21?

Hal ini terjadi karena model sudah mencapai titik plateau, yaitu kondisi di mana akurasi atau performa tidak lagi meningkat secara konsisten. Pada keadaan tersebut, pelatihan lanjut tidak lagi memberi manfaat yang berarti dan malah berisiko menimbulkan overfitting.

Secara umum, alasan berhenti pada epoch sekitar 21 adalah:

- tidak ada peningkatan signifikan pada metrik evaluasi validasi,
- loss sudah stabil dan tidak menurun lagi secara berarti,
- model mencapai kondisi yang cukup baik untuk saat itu,
- penggunaan early stopping menjaga agar training tidak terus berjalan tanpa manfaat.

### Kenapa early stopping penting?

Early stopping memiliki beberapa manfaat, yaitu:

- menghemat waktu pelatihan,
- mencegah overfitting,
- menjaga model tetap berada pada kondisi terbaiknya,
- menghindari pemborosan komputasi ketika model sudah stabil.

Jadi, walaupun rencana awalnya adalah 100 epoch, model bisa berhenti lebih awal jika sudah mencapai kondisi yang stabil. Dalam konteks ini, epoch 21 bukan berarti gagal, melainkan proses yang masuk akal untuk menjaga kualitas model dan mencegah pelatihan yang tidak efisien.

## 8. Hasil Training dan Pemilihan Model Terbaik

Setiap eksperimen pada file 06_latih_bandingkan.py menghasilkan bobot model yang disimpan dalam folder `hasil_praktik/`. Terdapat dua checkpoint utama yang dihasilkan, yaitu:

- `best.pt` : model dengan performa terbaik berdasarkan metrik validasi. Biasanya dipilih dari epoch yang memiliki nilai mAP50-95 tertinggi.
- `last.pt` : model hasil dari epoch terakhir pada proses training. Model ini mungkin belum tentu paling baik, tetapi tetap merepresentasikan hasil akhir pelatihan.

Pada skrip tersebut, model terbaik dipilih dengan logika berikut:

```python
terbaik = max(hasil, key=lambda r: r["mAP50-95"])
```

Artinya, eksperimen yang memiliki mAP50-95 paling tinggi pada data validasi dinyatakan sebagai model terbaik. Setelah itu, model terbaik tersebut diuji sekali lagi pada data test untuk mendapatkan hasil evaluasi akhir.

Dalam praktiknya, hasil model dapat dilihat dari folder seperti:

```text
hasil_praktik/
├── E1_pretrained/weights/best.pt
├── E1_pretrained/weights/last.pt
├── E2_scratch/weights/best.pt
├── E2_scratch/weights/last.pt
├── E3_tuning/weights/best.pt
├── E3_tuning/weights/last.pt
```

File `.pt` adalah file biner hasil training, bukan file teks. Oleh karena itu, file tersebut tidak bisa dibuka seperti file `.txt` atau `.csv`. File ini harus dipakai oleh YOLO untuk proses inferensi, validasi, atau evaluasi model.

## 9. Indikator Hasil Training yang Baik

Hasil training yang baik biasanya ditandai dengan:

- loss train menurun secara konsisten,
- loss validasi juga menurun atau setidaknya stabil,
- precision dan recall meningkat,
- mAP50 dan mAP50-95 meningkat,
- tidak muncul overfitting pada grafik evaluasi.

Jika grafik loss turun terus, tetapi mAP tidak naik, maka kemungkinan ada masalah pada kualitas data atau anotasi. Sedangkan jika loss validasi naik padahal loss train turun, maka model kemungkinan sedang overfitting.

## 10. Persiapan Lingkungan

Untuk menjalankan project ini, diperlukan environment Python dengan package berikut:

- OpenCV
- Ultralytics
- NumPy
- PyYAML

Disarankan untuk menggunakan virtual environment agar instalasi package tidak mengganggu lingkungan sistem.

## 11. Cara Menjalankan Project

### 1. Buat virtual environment

```bash
cd /home/aldi-faber/Downloads/YOLO
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependency

```bash
pip install ultralytics opencv-python
```

### 3. Jalankan proses data

```bash
python YOLO/00_ambil_subset_coco.py --kelas fork spoon cup --jumlah 200 --latar 0.1 --out dataset_yolo
```

### 4. Cek anotasi

```bash
python YOLO/01_cek_anotasi.py dataset_yolo
```

### 5. Train model

```bash
python YOLO/06_latih_bandingkan.py --nim 4222401002 --nama "Aldi Faber Sulaeman Pakpahan" --epochs 100 --patience 20
```

## 12. Kesimpulan

Project ini merupakan pipeline lengkap untuk membangun model deteksi objek YOLO pada kelas fork, spoon, dan cup. Dengan pendekatan subset COCO, pengecekan anotasi, pembagian dataset, serta eksperimen training, proyek ini mampu menghasilkan model dengan performa yang dapat dievaluasi secara objektif.

Penting untuk dipahami bahwa early stopping bukanlah kegagalan, melainkan strategi yang tepat untuk menghentikan pelatihan ketika model tidak lagi mengalami perbaikan. Dengan demikian, hasil yang diperoleh lebih efisien, lebih stabil, dan lebih aman dari risiko overfitting.

---

Dokumen ini dibuat untuk menjadi panduan pengelolaan project YOLO dan menjelaskan alasan teknis di balik keputusan pelatihan dengan jumlah epoch yang lebih pendek dari rencana semula.
