# Live eval (RAG answers judged by the local LLM)

- Date: 2026-11-24
- LLM (answers and judge): `gemma2:9b` via Ollama, temperature 0
- Embedding model: `intfloat/multilingual-e5-small`
- top_k=6, min_score=0.75
- Questions: 24 answerable, 4 off-topic; errors: 0

| metric | value | over |
| --- | --- | --- |
| answered rate | 95.8% | answerable questions |
| grounded (judge) | 91.3% | answered questions |
| citation-page accuracy | 100.0% | answered answerable questions (cited ∩ expected ≠ ∅) |
| correct pages (judge) | 91.3% | answered answerable questions |
| not-found accuracy | 100.0% | off-topic questions |
| avg latency of `ask` | 28.7 s | all questions |
| p95 latency of `ask` | 40.7 s | all questions |

| # | question | expected | found | cited | grounded | pages ok | latency | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Bir programın bir işi öğrendiğini söyleyebilmek için hangi üç unsura bakılır? | 1 | yes | 1 | yes | yes | 32.0 s |  |
| 2 | Derin öğrenmenin 2012'den sonra hızla yükselmesini sağlayan etkenler nelerdir? | 1 | yes | 1 | yes | yes | 28.7 s |  |
| 3 | Why could the single-layer perceptron not solve the XOR problem, and who pointed this out? | 1 | yes | 1 | yes | yes | 27.3 s |  |
| 4 | Sınıflandırma ile regresyon arasındaki fark nedir? | 2 | yes | 2 | yes | yes | 27.0 s |  |
| 5 | Öğrenme oranını çok yüksek seçersek eğitimde ne olur? | 2 | yes | 2 | yes | yes | 22.3 s |  |
| 6 | How does the k-nearest neighbors method decide the class of a new example? | 2 | yes | 2 | yes | yes | 26.7 s |  |
| 7 | k-means algoritmasında merkezler nasıl güncellenir ve algoritma ne zaman durur? | 3 | yes | 3 | yes | yes | 30.9 s |  |
| 8 | Kaç küme kullanacağımıza karar vermek için hangi yöntemlerden yararlanabiliriz? | 3 | yes | 3 | yes | yes | 30.2 s |  |
| 9 | Yüksek boyutlu veriyi az sayıda bileşene indirmek için hangi teknik kullanılır? | 3 | yes | 3 | yes | yes | 27.2 s |  |
| 10 | Eğitim hatası düşükken test hatasının yüksek olması neyi gösterir? | 4 | yes | 4 | yes | yes | 29.8 s |  |
| 11 | Lasso ile ridge cezası arasındaki temel fark nedir? | 4 | yes | 4 | yes | yes | 29.9 s |  |
| 12 | What is the bias-variance trade-off? | 4 | no | - | - | - | 29.5 s |  |
| 13 | Dengesiz veri kümelerinde doğruluk metriği neden yanıltıcı olabilir? | 5 | yes | 5 | yes | yes | 31.9 s |  |
| 14 | Kanser taramasında hangi metriğe öncelik verilmeli ve neden? | 5 | yes | 5 | yes | yes | 30.0 s |  |
| 15 | What does an AUC value of 0.5 tell us about a classifier? | 5 | yes | 5 | yes | yes | 41.9 s |  |
| 16 | Bir karar ağacı düğümde hangi bölünmeyi seçeceğine nasıl karar verir? | 6 | yes | 6 | no | yes | 29.3 s |  |
| 17 | Rastgele ormanı sıradan torbalamadan ayıran ek rastgelelik nedir? | 6 | yes | 6 | no | yes | 33.5 s |  |
| 18 | Boosting yönteminde ağaçlar nasıl eğitilir? | 6 | yes | 6 | yes | yes | 29.8 s |  |
| 19 | Aktivasyon fonksiyonu kullanılmasaydı bir sinir ağı neye dönüşürdü? | 7 | yes | 7 | yes | yes | 37.5 s |  |
| 20 | Geri yayılım sırasında gradyanlar nasıl hesaplanır? | 7 | yes | 7 | yes | yes | 40.7 s |  |
| 21 | Which network architecture is the foundation of today's large language models? | 7 | yes | 7 | yes | yes | 38.6 s |  |
| 22 | Cinsiyet sütununu veriden silmek ayrımcılığı neden tamamen önlemez? | 8 | yes | 8 | yes | yes | 36.4 s |  |
| 23 | Fırsat eşitliği ile demografik eşitlik arasındaki fark nedir? | 8 | yes | 8 | yes | yes | 38.0 s |  |
| 24 | Give an example of sampling bias in a face recognition system. | 8 | yes | 8 | yes | yes | 38.4 s |  |
| 25 | 2018 Dünya Kupası finalini hangi takım kazandı? | off-topic | no | - | - | - | 0.5 s |  |
| 26 | Mercimek çorbası nasıl yapılır? | off-topic | no | - | - | - | 36.4 s |  |
| 27 | Should I buy Apple stock before the next earnings report? | off-topic | no | - | - | - | 0.5 s |  |
| 28 | Avustralya'nın başkenti neresidir? | off-topic | no | - | - | - | 0.1 s |  |
