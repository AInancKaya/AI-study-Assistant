# 04 – Embedding (metni vektöre çevirmek)

Bu not, RAG hattının "anlam" katmanını anlatıyor: chunk'ları ve soruları sayı
vektörlerine çevirip benzerliklerini ölçmek. Ayrıca testlerde gerçek modelin yerine
geçen sahte (fake) sınıfları da anlatıyor.

---

## Ne yaptık?

- **`src/study_assistant/embeddings.py`** yazdık:
  - `Embedder` bir `Protocol`: `embed_passages(texts)` ve `embed_query(text)` metotları
    olan her sınıf "embedder" sayılır. Uygulamanın geri kalanı yalnızca bu arayüzü bilir.
  - `E5Embedder(model_name)`: `intfloat/multilingual-e5-small` modelini
    `sentence-transformers` ile çalıştırır. Model **ilk kullanımda** yüklenir
    (`_get_model`), parçalar `"passage: "`, sorular `"query: "` önekini alır,
    `normalize_embeddings=True`, `batch_size=32`, `show_progress_bar=False`.
  - Boş liste gelirse `embed_passages([])` modeli hiç yüklemeden `[]` döner.
  - Çıktı numpy dizisi değil, düz Python `list[list[float]]`.
- **`tests/fakes.py`** yazdık (sonraki testlerin hepsi bunu kullanacak):
  - `HashEmbedder(dim=256)`: model olmadan, kelimeleri `hashlib.md5` ile bir "kovaya"
    atayan, deterministik bir embedder.
  - `FakeLLM(responses, check_error=None)`: sıradaki hazır cevabı döndüren sahte LLM;
    her çağrıyı `self.calls` listesine kaydeder.
  - `cos(a, b)` ve `norm(v)`: testlerde kullanılan küçük vektör yardımcıları.
- **`tests/test_embeddings.py`**: 14 hızlı test + 1 `@pytest.mark.slow` test (gerçek
  modeli indirip çalıştırır; varsayılan `uv run pytest` bunu çalıştırmaz).

## Neden?

**Neden embedding?** Anahtar kelime araması "fotosentez nedir?" sorusunu, içinde
"bitkiler ışıkla besin üretir" yazan paragrafla eşleştiremez; ortak kelime azdır.
Embedding modeli metnin *anlamını* bir vektöre sıkıştırır; anlamca yakın metinlerin
vektörleri de birbirine yakın olur. Bu sayede soruya en alakalı chunk'ları bulabiliriz.

**Neden yerel (local) model?** Projenin kuralı: ücretli API yok, API anahtarı yok.
`multilingual-e5-small` yaklaşık 470 MB, CPU'da çalışır, bir kere indirilir ve
Hugging Face önbelleğinde kalır. Notlar bilgisayardan hiç çıkmaz (gizlilik).

**Neden çok dilli (multilingual)?** Ders notları Türkçe. Sadece İngilizce eğitilmiş bir
model "ışık", "üretmesi" gibi kelimeleri iyi temsil edemez. e5'in çok dilli sürümü
Türkçe dahil 100'den fazla dilde eğitildi.

**Neden lazy (tembel) yükleme?** `sentence_transformers` içe aktarılınca torch da
yüklenir; bu saniyeler sürer ve bazen uyarı basar. Import'u `_get_model` içine
koyduk; böylece birim testleri torch'a hiç dokunmaz, çıktı temiz kalır, API de ilk
istekten önce hızlı açılır. `test_importing_embeddings_does_not_import_torch` bunu
ayrı bir Python sürecinde doğrular.

**Neden sahte sınıflar?** Birim testleri ağa, Ollama'ya ve gerçek modele dokunmamalı.
`HashEmbedder` milisaniyede çalışır ve her çalıştırmada aynı sonucu verir. `FakeLLM`
ile "LLM bozuk JSON döndürdü", "Ollama kapalı" gibi senaryoları kolayca üretiriz.

## Nasıl çalışıyor?

**E5Embedder akışı:**

1. `embed_query("Fotosentez nedir?")` → metin `"query: Fotosentez nedir?"` olur.
2. `_encode` → `_get_model()` modeli gerekirse yükler, sonra `model.encode(...)` çağrılır.
3. Model 384 boyutlu bir vektör üretir; `normalize_embeddings=True` sayesinde vektörün
   uzunluğu (L2 normu) 1'dir.
4. numpy değerleri `float`'a çevrilip liste olarak döner (ChromaDB ve JSON için rahat).

**Neden önek?** e5 modelleri eğitilirken soru metinleri `query:`, belge metinleri
`passage:` önekiyle verildi. Model bu rolleri ayırt etmeyi öğrendi; önek unutulursa
arama kalitesi belirgin şekilde düşer. Bu yüzden önekleri çağıranın değil, sınıfın
kendisinin eklemesini sağladık: unutmak mümkün değil.

**Kosinüs benzerliği:** İki vektör arasındaki açının kosinüsü:

```
cos(a, b) = (a · b) / (|a| · |b|)
```

1'e yakınsa aynı yöne bakarlar (anlamca yakın), 0'a yakınsa ilgisizdirler. Vektörler
zaten normalize ise `|a| = |b| = 1` olur ve kosinüs sadece nokta çarpımıdır; hesap ucuzlar.
`config.py`'deki `min_score` (0.75) eşiği bu skora uygulanacak (Task 7).

**HashEmbedder akışı:**

1. `turkish_lower`: önce `İ→i`, `I→ı`, sonra `.lower()`. Python'un `"I".lower()`
   sonucu `"i"`'dir; Türkçede ise `"ı"` olmalı. `"İ".lower()` ise `"i̇"` (noktalı
   birleşik karakter) üretir, bu da kelimeyi bozar. Bu yüzden ikisini önce elle çeviriyoruz.
2. `\w+` ile kelimelere ayır.
3. Her kelimenin md5 özetinden bir kova numarası çıkar (`% dim`), o kovayı 1 artır.
4. Vektörü L2 ile normalize et. Boş metin sıfır vektör döner.

Python'un yerleşik `hash()` fonksiyonu her süreçte farklı tohum (salt) kullanır; md5
ise her zaman aynı sonucu verir. Testlerin tekrarlanabilir olması için md5 seçtik.

**FakeLLM:** `complete_json(system, user, schema)` çağrısı önce kaydedilir, sonra kuyruğun
başı alınır. O bir `Exception` ise fırlatılır; istenen `schema` tipinde değilse
`AssertionError` (yanlış kablolamayı yakalar); kuyruk boşsa
`AssertionError("FakeLLM: no more responses")`. `check()` normalde `None` döner;
`check_error` verildiyse onu fırlatır (Ollama kapalı senaryosu, Task 10).

## Önemli kavramlar

- **Embedding:** Metni sabit uzunlukta bir sayı vektörüne çeviren temsil.
- **Boyut (dimension):** Vektördeki sayı adedi; `multilingual-e5-small` için 384.
- **L2 normalizasyonu:** Vektörü uzunluğu 1 olacak şekilde ölçeklemek.
- **Kosinüs benzerliği:** Yön benzerliği; normalize vektörlerde nokta çarpımına eşit.
- **Asimetrik arama:** Kısa soru ile uzun paragrafın eşleştirilmesi; e5 bunu
  `query:`/`passage:` önekleriyle çözer.
- **Batch (toplu) işleme:** `batch_size=32` metni modele 32'şerli gruplar halinde verir;
  bellek ile hız arasında denge.
- **Lazy loading:** Pahalı bir kaynağı ancak gerçekten gerektiğinde yüklemek.
- **Protocol (yapısal tipleme):** Kalıtım gerektirmeden "şu metotlara sahip olan"
  her nesneyi kabul etmek; `E5Embedder` ve `HashEmbedder` ortak sınıftan türemez.
- **Test dublörü (fake):** Gerçek bağımlılığın hızlı, deterministik, sahte sürümü.
- **Pytest marker:** `slow` işaretli testler `pyproject.toml`'daki
  `-m 'not slow and not live'` ile varsayılan çalıştırmadan çıkarılır;
  `uv run pytest -m slow` ile ayrıca çalıştırılır.

## Mülakatta sorulabilecek sorular

1. **Embedding nedir, RAG'de ne işe yarar?**
   Metnin anlamını vektöre çevirir; soru vektörüne en yakın chunk vektörleri bulunup
   LLM'e bağlam olarak verilir.
2. **Kosinüs benzerliği ile Öklid mesafesi arasındaki fark nedir?**
   Kosinüs yöne bakar, uzunluğu yok sayar. Normalize vektörlerde ikisi aynı sıralamayı
   verir: `|a-b|² = 2 - 2·cos(a, b)`.
3. **e5 modellerinde `query:` ve `passage:` önekleri neden var?**
   Model asimetrik arama için bu öneklerle eğitildi; önek olmadan kalite düşer.
4. **Neden vektörleri normalize ediyorsunuz?**
   Benzerlik nokta çarpımına iner, skorlar 0–1 civarında karşılaştırılabilir olur ve
   `min_score` gibi sabit bir eşik anlamlı hale gelir.
5. **Embedding modelini değiştirirseniz ne olur?**
   Eski vektörler yeni modelle uyumsuzdur; tüm belgeler yeniden embed edilmeli.
   Bu yüzden model adı `config.py`'de (`embedding_model`) tek yerde tutulur.
6. **Birim testlerde gerçek modeli neden kullanmıyorsunuz?**
   Yavaş, ağ ister (ilk indirme), sonuçları kırılgan olabilir. `HashEmbedder`
   deterministik ve hızlıdır; gerçek model ayrı bir `slow` testle doğrulanır.
7. **Türkçe metinde küçük harfe çevirmenin tuzağı nedir?**
   Python `I`'yı `i`'ye çevirir; Türkçede `ı` olmalı. `İ` ise birleşik noktalı karaktere
   dönüşür. Çözüm: `İ→i`, `I→ı` dönüşümünü `.lower()`'dan önce yapmak.
8. **Lazy loading'in dezavantajı nedir?**
   İlk istek yavaş olur (model yükleme). Gerekirse uygulama açılışında bir "ısınma"
   çağrısı yapılabilir.
