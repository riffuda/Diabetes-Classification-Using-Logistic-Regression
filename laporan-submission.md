
# Prediksi Risiko Diabetes Menggunakan Logistic Regression dan Random Forest

Author: Rifda Qurrotul 'Ain 
Platform: Dicoding Submission – Proyek Machine Learning Terapan  
Domain: Kesehatan  
Metode: Klasifikasi Biner dengan Logistic Regression dan Random Forest

## 1. Domain Proyek

### Latar Belakang

Diabetes merupakan penyakit kronis yang menjadi salah satu penyebab kematian terbesar di dunia. Penyakit ini seringkali tidak terdeteksi pada tahap awal karena gejalanya yang cenderung samar. Oleh karena itu, deteksi dini terhadap risiko diabetes menjadi sangat penting untuk mencegah komplikasi jangka panjang. Teknologi machine learning kini menjadi pendekatan yang menjanjikan untuk membantu proses deteksi dini berdasarkan data riwayat kesehatan dan kebiasaan gaya hidup pasien.

Model prediktif berbasis machine learning telah banyak diteliti dan terbukti efektif dalam mengidentifikasi potensi risiko diabetes secara akurat dan cepat. Dengan memanfaatkan data yang tersedia, model ini dapat digunakan sebagai sistem pendukung keputusan dalam bidang kesehatan.

### Mengapa Masalah Ini Perlu Diselesaikan?

* Membantu proses diagnosis lebih awal, sehingga pasien dapat melakukan intervensi sejak dini.
* Memberikan dukungan bagi tenaga medis untuk pengambilan keputusan berbasis data.
* Efisiensi dalam penanganan dan pencegahan komplikasi diabetes di tingkat populasi.

### Referensi

* Kaur, H. and Kumari, V. (2022), "Predictive modelling and analytics for diabetes using a machine learning approach", *Applied Computing and Informatics*, Vol. 18 No. 1/2, pp. 90–100. [https://doi.org/10.1016/j.aci.2018.12.004](https://doi.org/10.1016/j.aci.2018.12.004)
* Modak, S.K.S., Jha, V.K. (2024). "Diabetes prediction model using machine learning techniques." *Multimedia Tools and Applications*, 83, 38523–38549. [https://doi.org/10.1007/s11042-023-16745-4](https://doi.org/10.1007/s11042-023-16745-4)

## 2. Business Understanding

### Problem Statement

Bagaimana cara memprediksi risiko diabetes berdasarkan data riwayat dan kebiasaan gaya hidup pasien?

### Goals

Membangun model klasifikasi penyakit diabetes menggunakan machine learning dengan akurasi tinggi untuk membantu deteksi dini dan pencegahan penyakit secara lebih cepat dan efisien.

### Solution Statement

Untuk mencapai tujuan di atas, dua pendekatan model machine learning digunakan dan dibandingkan:

1. **Logistic Regression (LR)**

   * Model statistik klasik yang banyak digunakan dalam klasifikasi biner.
   * Memiliki interpretabilitas tinggi dan efisien untuk baseline.
   * Memberikan insight yang mudah dipahami terhadap pengaruh masing-masing fitur terhadap risiko diabetes.

2. **Random Forest Classifier (RF)**

   * Model ensemble berbasis decision tree yang lebih kompleks.
   * Dapat menangani data dengan hubungan non-linear dan fitur yang saling berinteraksi.
   * Diharapkan mampu memberikan performa yang lebih baik dengan tuning parameter dan seleksi fitur.

## 3. Data Understanding

### Dataset
* Nama: `diabetes_data.csv`
* Sumber: [Diabetes, Hypertension and Stroke Prediction](https://www.kaggle.com/datasets/prosperchuks/health-dataset)
* Jumlah sampel: 70.692 baris
* Tipe data: Numerik kategorikal dan numerik kontinu
* Target klasifikasi: `Diabetes` (0 = tidak, 1 = ya)

### Deskripsi Dataset
Dataset ini berisi data survei kesehatan masyarakat yang mencakup berbagai aspek seperti usia, indeks massa tubuh (BMI), tekanan darah tinggi, kebiasaan merokok, konsumsi buah/sayur, dan aktivitas fisik. Tujuannya adalah untuk memprediksi apakah seseorang berisiko menderita diabetes berdasarkan kombinasi variabel tersebut.

### Statistik Dataset
- Jumlah data: 70.692 baris, 18 kolom (termasuk target)
- Tidak terdapat missing value
- Distribusi target:
  - Kelas 0 (tidak diabetes): 50.0%
  - Kelas 1 (diabetes): 50.0% 
- Rata-rata BMI: 29.8
- Usia terbanyak: kategori 60–64 tahun (kode 9)
- 56% responden memiliki tekanan darah tinggi
- 52% memiliki kolesterol tinggi

### Fitur Dataset:

| Fitur                   | Deskripsi                                                             |
|-------------------------|----------------------------------------------------------------------|
| Age                    | Kategori usia (1 = 18–24, …, 13 = ≥80)                               |
| Sex                    | Jenis kelamin (0 = perempuan, 1 = laki-laki)                         |
| HighChol               | Kolesterol tinggi (1 = ya, 0 = tidak)                                |
| CholCheck              | Cek kolesterol dalam 5 tahun terakhir (1 = ya, 0 = tidak)            |
| BMI                    | Indeks massa tubuh                                                   |
| Smoker                 | Pernah merokok ≥100 batang seumur hidup (1 = ya, 0 = tidak)          |
| HeartDiseaseorAttack   | Riwayat serangan jantung (1 = ya, 0 = tidak)                         |
| PhysActivity           | Aktivitas fisik dalam 30 hari terakhir (1 = ya, 0 = tidak)           |
| Fruits                 | Konsumsi buah harian (1 = ya, 0 = tidak)                             |
| Veggies                | Konsumsi sayur harian (1 = ya, 0 = tidak)                            |
| HvyAlcoholConsump      | Konsumsi alkohol berat (1 = ya, 0 = tidak)                           |
| GenHlth                | Penilaian kesehatan umum (1 = sangat baik, 5 = sangat buruk)         |
| MentHlth               | Jumlah hari dengan masalah mental dalam 30 hari terakhir             |
| PhysHlth               | Jumlah hari dengan masalah fisik dalam 30 hari terakhir              |
| DiffWalk               | Kesulitan berjalan (1 = ya, 0 = tidak)                               |
| Stroke                 | Riwayat stroke (1 = ya, 0 = tidak)                                   |
| HighBP                 | Tekanan darah tinggi (1 = ya, 0 = tidak)                             |
| Diabetes               | Target klasifikasi (1 = diabetes, 0 = tidak)                         |

### Exploratory Data Analysis (EDA)

Langkah-langkah EDA yang dilakukan antara lain:
- Visualisasi distribusi target (pie chart dan bar chart)
- Distribusi kategori usia dan jenis kelamin
- Korelasi antar fitur numerik dengan target `Diabetes`
- Statistik deskriptif (mean, median, modus) untuk fitur-fitur utama

## 4. Data Preparation

### Pembersihan Data
- Dataset sudah tidak memiliki missing value, sehingga tidak diperlukan imputasi.
- Duplikasi data tidak ditemukan.
- Beberapa nilai ekstrem ditemukan pada kolom `BMI`, `MentHlth`, dan `PhysHlth`, namun tetap dipertahankan karena dapat mencerminkan kondisi asli populasi.
- Semua data merupakan numerik, sehingga tidak diperlukan encoding tambahan.

### Proses Standarisasi dan Split Data
- Standarisasi dilakukan menggunakan `StandardScaler` agar semua fitur memiliki skala yang seragam (mean = 0, std = 1).
- Data dibagi menjadi training set dan testing set dengan rasio 70:30.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
```

## 5. Modeling

### Pendekatan Model

Permasalahan yang dihadapi dalam proyek ini merupakan klasifikasi biner, yaitu menentukan apakah seseorang berisiko mengidap diabetes berdasarkan atribut medis dan gaya hidup. Dua model machine learning digunakan untuk menyelesaikan masalah ini: Logistic Regression dan Random Forest.

### 1. Logistic Regression (LR)

* **Deskripsi:** Logistic Regression merupakan model statistik klasik yang digunakan untuk klasifikasi biner. Model ini memodelkan probabilitas kelas target dengan fungsi logistik dari kombinasi linier fitur.
* **Parameter:**

  * `class_weight='balanced'`: digunakan untuk menangani distribusi kelas yang tidak seimbang.
* **Kelebihan:**

  * Cepat dan efisien untuk dataset berukuran sedang.
  * Mudah diinterpretasikan, sehingga berguna dalam konteks medis.
* **Kekurangan:**

  * Kurang mampu menangkap hubungan non-linier antar fitur.
  * Sensitif terhadap multikolinearitas.

### Diagram Alur Model Regresi Logistik
<p align="center">
  <img src="https://zd-brightspot.s3.us-east-1.amazonaws.com/wp-content/uploads/2022/04/11040521/46-4-e1715636469361.png" alt="Diagram Alur Model Regresi Logistik" width="500"/>
</p>

### 2. Random Forest (RF)

* **Deskripsi:** Random Forest adalah algoritma ensemble yang menggabungkan banyak decision tree untuk meningkatkan akurasi dan mengurangi overfitting.
* **Parameter yang digunakan:**

  * `n_estimators=200`: jumlah pohon dalam ensemble.
  * `max_depth=15`: batas maksimum kedalaman pohon.
  * `min_samples_split=5`: minimum jumlah sampel untuk memisahkan node.
  * `min_samples_leaf=2`: minimum jumlah sampel dalam daun.
  * `class_weight='balanced'`: untuk menangani distribusi kelas tidak seimbang.
  * `random_state=50`: untuk memastikan replikasi hasil.
* **Kelebihan:**

  * Dapat menangani fitur non-linier dan interaksi kompleks antar variabel.
  * Lebih tahan terhadap overfitting dibandingkan decision tree tunggal.
* **Kekurangan:**

  * Kurang transparan dalam interpretasi dibandingkan model linear.
  * Komputasi lebih berat.

### Diagram Alur Model Random Forest
<p align="center">
  <img src="https://media.geeksforgeeks.org/wp-content/uploads/20250522115823647286/Random_Forest_Algorithm.webp" alt="Diagram Alur Model Random Forest" width="500"/>
</p>

### Pemilihan Model Terbaik

Setelah dilakukan pelatihan dan evaluasi terhadap kedua model, Random Forest menunjukkan performa yang lebih baik dari segi akurasi dan F1-score. Model ini mampu menangkap pola yang lebih kompleks dalam data, terutama interaksi antar fitur yang mungkin tidak linier.

Dengan pertimbangan tersebut, **Random Forest dipilih sebagai model terbaik** dalam proyek ini karena mampu memberikan hasil yang lebih akurat, meskipun dengan interpretabilitas yang lebih rendah dibanding Logistic Regression.

## 6. Evaluation

### Metrik Evaluasi:
Model dievaluasi menggunakan confusion matrix dan classification report untuk mengukur performa prediksi pada data uji.

Metrik yang digunakan:
- **Accuracy** — proporsi prediksi yang benar dari total prediksi
- **Precision** — ketepatan prediksi positif (diabetes)
- **Recall** — kemampuan model mengenali semua kasus diabetes
- **F1-Score** — rata-rata harmonis antara precision dan recall

### Confusion Matrix

Confusion matrix adalah tabel yang menunjukkan jumlah prediksi benar dan salah yang dibuat oleh model klasifikasi, dibagi ke dalam kategori:

* **True Positive (TP)**: Kasus diabetes yang berhasil diprediksi dengan benar.
* **True Negative (TN)**: Kasus tidak diabetes yang berhasil diprediksi dengan benar.
* **False Positive (FP)**: Kasus tidak diabetes yang salah diklasifikasikan sebagai diabetes (false alarm).
* **False Negative (FN)**: Kasus diabetes yang tidak berhasil dikenali (terlewatkan).

Confusion matrix membantu untuk melihat bagaimana kesalahan model tersebar di tiap kelas, dan menjadi dasar untuk menghitung precision, recall, dan F1-score.
Berikut adalah hasil evaluasi model:

### Logistic Regression:
| Kelas         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Tidak Diabetes| 0.76      | 0.72   | 0.74     | 10601   |
| Diabetes      | 0.74      | 0.77   | 0.75     | 10607   |
| **Accuracy**  |           |        | **0.75** | **21208** |
| Macro Avg     | 0.75      | 0.75   | 0.75     | 21208   |
| Weighted Avg  | 0.75      | 0.75   | 0.75     | 21208   |

### Random Forest:
| Kelas         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Tidak Diabetes| 0.78      | 0.70   | 0.74     | 10601   |
| Diabetes      | 0.73      | 0.80   | 0.76     | 10607   |
| **Accuracy**  |           |        | **0.75** | **21208** |
| Macro Avg     | 0.75      | 0.75   | 0.75     | 21208   |
| Weighted Avg  | 0.75      | 0.75   | 0.75     | 21208   |

Kedua model menunjukkan performa yang seimbang dengan akurasi sekitar 75%. Logistic Regression menghasilkan prediksi yang cukup seimbang antar kelas, sedangkan Random Forest memiliki recall yang lebih tinggi pada kelas diabetes, menjadikannya lebih efektif dalam mendeteksi kasus positif.

## 7. Kesimpulan

Model klasifikasi risiko diabetes berhasil dibangun menggunakan dua pendekatan: Logistic Regression dan Random Forest. Keduanya diuji pada data survei kesehatan dan menunjukkan hasil yang cukup baik.

Beberapa poin penting dari proyek ini:

- Kedua model mencapai **akurasi sekitar 75%**, cukup baik untuk klasifikasi biner.
- **Logistic Regression** memberikan hasil yang stabil dan seimbang antar kelas.
- **Random Forest** unggul dalam mendeteksi kasus diabetes dengan recall mencapai 80%.
- Evaluasi melalui confusion matrix menunjukkan bahwa kedua model memiliki keandalan dalam prediksi, dengan pendekatan berbeda.
- Model ini membuktikan bahwa data kesehatan dasar seperti tekanan darah, BMI, dan usia bisa digunakan untuk membangun sistem skrining awal diabetes yang cukup akurat.

### Pengembangan ke depan:
- Menambahkan fitur tambahan seperti riwayat keluarga, pola makan, dan jam tidur untuk meningkatkan prediksi.
- Melakukan tuning hyperparameter lebih lanjut untuk optimasi model.
- Mencoba model lain seperti XGBoost atau LightGBM untuk potensi peningkatan akurasi.

Model ini bisa dijadikan langkah awal untuk sistem deteksi dini berbasis data yang dapat digunakan di layanan kesehatan atau kampanye pencegahan.

