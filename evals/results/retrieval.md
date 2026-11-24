# Retrieval eval

- Date: 2026-11-24
- Embedding model: `intfloat/multilingual-e5-small`
- Notes: `evals/data/sample_notes.pdf`, chunk_size=1500, chunk_overlap=200, k=6

| metric | value |
| --- | --- |
| answerable questions (n) | 24 |
| hit@6 | 0.958 |
| MRR | 0.889 |
| lowest correct top-1 score | 0.786 |
| highest off-topic top-1 score | 0.787 |
| suggested min_score | no clean threshold |
| configured min_score | 0.75 |

| # | lang | question | expected | rank | top-1 page | top-1 score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | tr | Bir programın bir işi öğrendiğini söyleyebilmek için hangi üç unsura bakılır? | 1 | 1 | 1 | 0.844 |
| 2 | tr | Derin öğrenmenin 2012'den sonra hızla yükselmesini sağlayan etkenler nelerdir? | 1 | 1 | 1 | 0.850 |
| 3 | en | Why could the single-layer perceptron not solve the XOR problem, and who pointed this out? | 1 | 1 | 1 | 0.786 |
| 4 | tr | Sınıflandırma ile regresyon arasındaki fark nedir? | 2 | 1 | 2 | 0.866 |
| 5 | tr | Öğrenme oranını çok yüksek seçersek eğitimde ne olur? | 2 | 1 | 2 | 0.847 |
| 6 | en | How does the k-nearest neighbors method decide the class of a new example? | 2 | 2 | 3 | 0.811 |
| 7 | tr | k-means algoritmasında merkezler nasıl güncellenir ve algoritma ne zaman durur? | 3 | 1 | 3 | 0.867 |
| 8 | tr | Kaç küme kullanacağımıza karar vermek için hangi yöntemlerden yararlanabiliriz? | 3 | 1 | 3 | 0.852 |
| 9 | tr | Yüksek boyutlu veriyi az sayıda bileşene indirmek için hangi teknik kullanılır? | 3 | 1 | 3 | 0.861 |
| 10 | tr | Eğitim hatası düşükken test hatasının yüksek olması neyi gösterir? | 4 | 2 | 5 | 0.843 |
| 11 | tr | Lasso ile ridge cezası arasındaki temel fark nedir? | 4 | 1 | 4 | 0.822 |
| 12 | en | What is the bias-variance trade-off? | 4 | - | 7 | 0.782 |
| 13 | tr | Dengesiz veri kümelerinde doğruluk metriği neden yanıltıcı olabilir? | 5 | 1 | 5 | 0.857 |
| 14 | tr | Kanser taramasında hangi metriğe öncelik verilmeli ve neden? | 5 | 1 | 5 | 0.847 |
| 15 | en | What does an AUC value of 0.5 tell us about a classifier? | 5 | 1 | 5 | 0.802 |
| 16 | tr | Bir karar ağacı düğümde hangi bölünmeyi seçeceğine nasıl karar verir? | 6 | 1 | 6 | 0.876 |
| 17 | tr | Rastgele ormanı sıradan torbalamadan ayıran ek rastgelelik nedir? | 6 | 1 | 6 | 0.866 |
| 18 | tr | Boosting yönteminde ağaçlar nasıl eğitilir? | 6 | 1 | 6 | 0.885 |
| 19 | tr | Aktivasyon fonksiyonu kullanılmasaydı bir sinir ağı neye dönüşürdü? | 7 | 1 | 7 | 0.851 |
| 20 | tr | Geri yayılım sırasında gradyanlar nasıl hesaplanır? | 7 | 1 | 7 | 0.860 |
| 21 | en | Which network architecture is the foundation of today's large language models? | 7 | 1 | 7 | 0.791 |
| 22 | tr | Cinsiyet sütununu veriden silmek ayrımcılığı neden tamamen önlemez? | 8 | 1 | 8 | 0.820 |
| 23 | tr | Fırsat eşitliği ile demografik eşitlik arasındaki fark nedir? | 8 | 1 | 8 | 0.869 |
| 24 | en | Give an example of sampling bias in a face recognition system. | 8 | 3 | 3 | 0.795 |
| 25 | tr | 2018 Dünya Kupası finalini hangi takım kazandı? | off-topic | - | 5 | 0.740 |
| 26 | tr | Mercimek çorbası nasıl yapılır? | off-topic | - | 3 | 0.787 |
| 27 | en | Should I buy Apple stock before the next earnings report? | off-topic | - | 5 | 0.729 |
| 28 | tr | Avustralya'nın başkenti neresidir? | off-topic | - | 1 | 0.718 |
