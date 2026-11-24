# 12 – Değerlendirme (eval): Sistem gerçekten işe yarıyor mu?

Bu not, uygulamanın kalitesini **sayıyla** ölçtüğümüz görevi anlatıyor. Önceki görevlerde
birim testleri yazdık; onlar kodun doğru çalıştığını gösterir. Değerlendirme (eval) ise
sorunun başka bir yüzünü sorar: gerçek embedding modeli ve gerçek LLM ile sistem öğrenciye
**doğru sayfayı bulup doğru cevabı** veriyor mu?

---

## Ne yaptık?

- **`evals/data/sample_notes.md`**: kendimiz yazdığımız 8 sayfalık Türkçe ders notu
  ("Makine Öğrenmesine Giriş"). Her sayfa ayrı bir konu: tanım ve tarihçe, denetimli
  öğrenme, kümeleme, aşırı öğrenme, metrikler, karar ağaçları, sinir ağları, etik.
  İlk satırdaki lisans notu bir HTML yorumu (`<!-- ... -->`); `scripts/make_pdf.py`
  içindeki `split_pages` artık tek satırlık HTML yorumlarını atlıyor, böylece PDF temiz
  kalıyor. PDF şu komutla üretildi:
  `uv run python scripts/make_pdf.py evals/data/sample_notes.md evals/data/sample_notes.pdf`
- **`evals/data/gold.jsonl`**: "altın set". 24 cevaplanabilir soru (18 Türkçe, 6 İngilizce)
  ve 4 konu dışı soru (futbol, yemek tarifi, borsa, başkent). Her satır:
  `{"question": ..., "expected_pages": [4], "lang": "tr"}`. Konu dışı sorularda
  `expected_pages` boş liste. Sorular nottaki cümleleri kopyalamıyor, **başka kelimelerle**
  soruyor; yoksa arama sadece kelime eşleştirmesini test etmiş olurdu.
- **`evals/retrieval_eval.py`**:
  - `first_relevant_rank(hits, expected_pages)`: beklenen sayfadaki ilk parçanın 1'den
    başlayan sırası, yoksa `None`.
  - `evaluate(store, gold, k) -> dict`: `hit_at_k`, `mrr`, `n` ve eşik önerisi
    (`relevant_min_score`, `offtopic_max_score`, `suggested_min_score`).
  - `build_store(path, settings)`: gerçek `extract_pages` + `chunk_pages` + `E5Embedder`
    ile geçici bir `VectorStore` kurar.
  - CLI: `uv run python evals/retrieval_eval.py` → `evals/results/retrieval.md` yazar,
    hit@6 < 0.8 ise **çıkış kodu 1** döner (CI'ı kırar).
- **`evals/live_eval.py`**: her altın soruyu gerçek Ollama ile `rag.ask` üzerinden çalıştırır,
  süresini ölçer, cevabı aynı yerel modele "hakem" olarak değerlendirtir (`JudgeVerdict`)
  ve `evals/results/live_eval.md` yazar.
- **Testler**: `tests/test_retrieval_eval.py` (sahte bir store ile bilinen sıralamalar:
  sıralar [1, 2, None] → hit@6 = 2/3, MRR = 0.5), `tests/test_live_eval.py` (özet
  metriklerin ve p95 hesabının matematiği), `tests/test_make_pdf.py` (yorum satırı atlama).
- **CI**: `.github/workflows/ci.yml` içine `needs: test` ile ayrı bir `eval` işi ekledik.
  `~/.cache/huggingface` klasörünü model adına göre önbelleğe alıyor, sonra
  `uv run python evals/retrieval_eval.py` çalıştırıyor.
- **Makefile**: `make eval` ve `make live-eval` artık bu betikleri de çalıştırıyor.

## Neden?

- **"Bende çalıştı" yetmez.** Birim testlerinde `HashEmbedder` ve `FakeLLM` kullandık;
  bunlar gerçek modelin kalitesi hakkında hiçbir şey söylemez. Eval, gerçek modelle ölçer.
- **Değişiklikleri karşılaştırmak için.** `chunk_size`, `top_k`, embedding modeli veya
  prompt değişince sistem iyileşti mi kötüleşti mi? Sabit bir altın set ve sabit metrikler
  olmadan bunu ancak tahmin edebiliriz.
- **Eşikleri veriyle seçmek için.** `min_score` değerini Görev 7'de 0.75 olarak tahmin
  etmiştik. Eval, bu değerin gerçekten doğru ve alakasız soruları ayırıp ayırmadığını
  gösteriyor.
- **README'deki sayılar gerçek olmalı.** Sonuç dosyaları tarih ve model adıyla birlikte
  commit edildi; herkes aynı komutla tekrar üretebilir.

## Nasıl çalışıyor?

**Retrieval eval (LLM yok, ~15 saniye):**

1. `build_store` PDF'i okur, parçalara böler (varsayılan 1500/200 karakter → 16 parça),
   E5 ile gömer ve geçici bir Chroma klasörüne yazar.
2. Her soru için `store.search(question, k=6)` çağrılır.
3. Cevaplanabilir sorularda `first_relevant_rank` sırayı bulur. hit@k = sırası olan
   soruların oranı; MRR = `1/sıra` değerlerinin ortalaması (bulunamayan soru 0 sayılır).
4. Eşik önerisi: top-1 sonucu doğru sayfada olan sorular arasındaki **en düşük** top-1
   skoru ile konu dışı soruların **en yüksek** top-1 skorunun orta noktası. Konu dışı skor
   doğru skordan büyük veya eşitse temiz bir eşik yoktur; `suggested_min_score = None`.

**Bu projede çıkan sonuç** (`evals/results/retrieval.md`): hit@6 = 0.958, MRR = 0.889.
En düşük doğru top-1 skoru 0.786, en yüksek konu dışı skor 0.787 ("Mercimek çorbası nasıl
yapılır?"). Yani iki dağılım üst üste biniyor: **temiz bir eşik yok**. Bu yüzden
`config.py` içindeki `min_score = 0.75` değerini değiştirmedik. Eşik yalnızca bariz
alakasız soruları (ör. başkent sorusu, 0.718) LLM'e gitmeden eliyor; geri kalanını
`ANSWER_SYSTEM` içindeki "kaynakta yoksa `found=false`" kuralı yakalamak zorunda.
E5 skorlarının dar bir aralıkta (yaklaşık 0.7–0.9) toplandığına dikkat et: kosinüs
benzerliği burada mutlak bir "alaka yüzdesi" değil.

Tek kaçan soru İngilizce "What is the bias-variance trade-off?" oldu; doğru sayfa (4)
Türkçe "yanlılık–varyans ödünleşimi" diyor. Diller arası (cross-lingual) arama çoğu zaman
çalışsa da küçük modelde zayıf noktalar kalıyor. Notları sorulara göre değiştirmedik;
eval'i "kandırmak" ölçümü anlamsız yapar.

**Live eval (Ollama, birkaç dakika):**

1. `run_item` her soru için `ask(...)` çağırır ve `time.perf_counter()` ile süreyi ölçer.
2. Cevap verildiyse (`found=True`) aynı kaynaklar (`score >= min_score` olan parçalar)
   `format_sources` ile yeniden hazırlanır ve hakeme gönderilir: soru, kaynaklar, cevap,
   atıf yapılan sayfalar ve beklenen sayfalar.
3. Hakem `JUDGE_SYSTEM` talimatıyla, sıcaklık 0'da, `{"grounded": bool,
   "correct_pages": bool}` döner.
4. `summarize` metrikleri toplar: cevaplanma oranı, grounded yüzdesi, atıf-sayfa doğruluğu
   (atıf ∩ beklenen ≠ ∅), konu dışı sorularda "bulunamadı" doğruluğu, ortalama ve p95 süre.

**Bu projede çıkan sonuç** (`evals/results/live_eval.md`, `gemma2:9b`): cevaplanma oranı
%95.8 (24 sorudan 23'ü; kaçan, aramanın da kaçırdığı bias-variance sorusu), grounded %91.3
(hakem 2 cevabı "kaynakta olmayan ayrıntı var" diye işaretledi), atıf-sayfa doğruluğu %100,
konu dışı sorularda "bulunamadı" doğruluğu %100. Mercimek çorbası sorusu eşiği geçti
(0.787) ama LLM `found=false` dedi: ikinci savunma hattı çalıştı. Diğer 3 konu dışı soru
LLM'e hiç gitmedi (0.5 saniyenin altında). Ortalama `ask` süresi 28.7 s, p95 40.7 s
(Apple Silicon üzerinde yerel 9B model; hakem çağrıları bu süreye dahil değil).

## Önemli kavramlar

- **Altın set (gold set):** doğru cevabı insan tarafından önceden belirlenmiş soru listesi.
  Eval'in kalitesi bu setin kalitesinden yüksek olamaz; ilk iş beklenen sayfaların doğru
  olduğunu kontrol etmektir.
- **hit@k (recall@k):** doğru parça ilk k sonuç içinde mi? RAG için önemli, çünkü LLM'e
  k parça gidiyor; doğru parça k'nın içindeyse model cevabı görebilir.
- **MRR (Mean Reciprocal Rank):** doğru sonucun ne kadar yukarıda olduğunu ölçer. Sıra 1 →
  1.0, sıra 2 → 0.5, sıra 3 → 0.33, bulunamadı → 0.
- **Eşik ayarlama:** iki skor dağılımını (alakalı / alakasız) karşılaştırıp aralarına bir
  çizgi çekmek. Dağılımlar örtüşüyorsa hiçbir eşik ikisini birden mükemmel ayıramaz.
- **LLM-as-judge (hakem olarak LLM):** serbest metin cevapları ölçmek için bir LLM'e
  puanlatmak. Ucuz ve hızlı, ama hakem de hata yapabilir; aynı modelin kendi cevabını
  yargılaması iyimser sonuç verebilir (self-preference bias). Bu yüzden sayfa doğruluğunu
  ayrıca **deterministik** olarak da (küme kesişimi) hesaplıyoruz.
- **Groundedness (temellendirilmişlik):** cevaptaki her iddianın kaynaklarda bulunması.
- **p95 gecikme:** isteklerin %95'inin bu süreden kısa sürdüğü değer; ortalama, birkaç çok
  yavaş isteği gizleyebilir.
- **Regresyon kapısı (regression gate):** CI'da metrik eşiğin altına düşerse derlemeyi
  kırmak (`MIN_HIT_AT_K = 0.8`).

## Mülakatta sorulabilecek sorular

1. **Birim testi ile eval arasındaki fark nedir?**
   Birim testi deterministik davranışı (doğru/yanlış) kontrol eder; eval, gerçek
   modellerle kaliteyi bir metrikle ölçer ve sonuç bir oran olur.
2. **hit@k ve MRR neyi ölçer, hangisi ne zaman yetersiz kalır?**
   hit@k doğru parçanın ilk k içinde olup olmadığını, MRR ise ne kadar yukarıda olduğunu
   ölçer. hit@k sıralamayı umursamaz; MRR sadece ilk doğru sonuca bakar.
3. **Altın setteki sorular neden notlardaki cümleleri kopyalamamalı?**
   Kopyalanmış soru, anlamsal aramayı değil kelime örtüşmesini test eder ve sonucu
   olduğundan iyi gösterir.
4. **`min_score` eşiğini nasıl seçerdin? Temiz bir eşik yoksa ne yaparsın?**
   Alakalı ve alakasız top-1 skorlarının dağılımına bakarım. Örtüşüyorsa eşiği muhafazakâr
   tutar, ikinci savunma hattını (LLM'in `found=false` demesi) kullanır ve daha iyi bir
   embedding modeli veya reranker denerim.
5. **LLM-as-judge'ın zayıf yönleri nelerdir, nasıl azaltırsın?**
   Hakem yanılabilir, uzun cevapları ve kendi üslubunu kayırabilir. Sıcaklık 0, yapısal
   çıktı, net kriterler, mümkün olan yerde deterministik kontroller ve küçük bir örneği
   insanla karşılaştırmak işe yarar.
6. **Eval sonuçlarını neden tarih ve model adıyla kaydettik?**
   Sayılar model, ayar ve veri değişince değişir; hangi koşulda ölçüldüğü bilinmeyen bir
   sayı tekrarlanamaz ve karşılaştırılamaz.
