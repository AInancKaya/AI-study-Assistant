# 05 – Vektör veritabanı (chunk'ları saklamak ve aramak)

Bu not, RAG hattının "hafıza" katmanını anlatıyor: embedding'leri çıkarılmış chunk'ları
diske kaydetmek ve bir soruya en çok benzeyen chunk'ları sayfa numaralarıyla birlikte
geri bulmak.

---

## Ne yaptık?

- **`src/study_assistant/store.py`** yazdık. İçinde tek bir sınıf var: `VectorStore`.
  - `VectorStore(path, embedder, collection="chunks")`: `chromadb.PersistentClient` ile
    `path` klasöründe kalıcı bir Chroma veritabanı açar. Koleksiyon
    `metadata={"hnsw:space": "cosine"}` ile oluşturulur. Telemetri kapalıdır
    (`Settings(anonymized_telemetry=False)`).
  - `add_document(chunks, pages)`: bir belgenin bütün chunk'larını ekler ve bir
    `DocumentInfo` döner. Her chunk'a şu metadata yazılır:
    `doc_id, filename, page, chunk_idx, doc_pages`.
  - `has_document(doc_id)`: belge kayıtlı mı?
  - `search(query, k, doc_id=None)`: soruya en benzer `k` chunk'ı `RetrievedChunk`
    listesi olarak döner; `score = 1 - distance`, büyükten küçüğe sıralı.
  - `get_chunks(doc_id, pages=None)`: bir belgenin chunk'larını (istenirse sadece
    bazı sayfalarınkini) `chunk_idx` sırasıyla döner. Quiz üretimi bunu kullanacak.
  - `list_documents()`: kayıtlı belgeleri dosya adına göre sıralı döner.
  - `delete_document(doc_id)`: belgeyi siler; yoksa `False` döner.
- **`tests/test_store.py`**: 12 hızlı test. Hepsi `tmp_path` (her teste ayrı geçici
  klasör) ve `tests/fakes.py` içindeki `HashEmbedder` kullanır; gerçek model yüklenmez.

## Neden?

- **Embedding'leri her seferinde yeniden hesaplamak pahalı.** 300 sayfalık bir PDF'in
  chunk'larını bir kez gömüp diske yazıyoruz; uygulama yeniden başlasa da duruyorlar
  (`test_persists_across_instances` bunu kontrol ediyor).
- **Benzerlik araması için özel bir yapı gerekir.** Binlerce vektörle tek tek kosinüs
  hesaplamak yerine Chroma bir **HNSW** indeksi kullanır ve hızlıca yaklaşık en yakın
  komşuları bulur.
- **Atıf (citation) için sayfa numarası şart.** Cevapta "bkz. sayfa 2" diyebilmek için
  her chunk'ın `page` bilgisini metadata olarak saklıyoruz.
- **Neden Chroma?** Ücretsiz, yerel çalışır, sunucu gerektirmez (`PersistentClient` bir
  klasöre yazar), Python'dan kolay kullanılır. Projenin "ücretli API yok" kuralına uyar.
- **Neden embedding'i kendimiz veriyoruz?** Chroma kendi varsayılan modelini indirmeye
  çalışabilir. Biz her zaman `embedder.embed_passages(...)` ve `embedder.embed_query(...)`
  sonucunu açıkça geçiriyoruz. Böylece testlerde `HashEmbedder`, gerçekte `E5Embedder`
  kullanılır ve ağ bağlantısı hiç gerekmez.

## Nasıl çalışıyor?

### Ekleme (`add_document`)

1. `_single_doc_id(chunks)` kontrol eder: liste boşsa ya da chunk'lar farklı
   `doc_id`'lere aitse `ValueError` fırlatır. Bir çağrı = bir belge.
2. Tüm chunk metinleri tek seferde `embed_passages` ile vektöre çevrilir.
3. Aynı `doc_id` daha önce eklenmişse önce silinir (`delete_document`). Yani aynı PDF'i
   iki kez yüklemek chunk sayısını ikiye katlamaz, eskisinin yerine geçer.
4. `collection.add(ids=..., embeddings=..., documents=..., metadatas=...)` çağrılır.
   Chunk `id`'si chunking'den gelir: `"{doc_id}:{page}:{chunk_idx}"`.

### Arama (`search`)

```python
n_results = min(k, self._collection.count())
if n_results <= 0:
    return []
```

- Chroma, koleksiyondaki kayıt sayısından fazla sonuç istenirse sorun çıkarabilir; bu
  yüzden `k` değerini kayıt sayısıyla sınırlıyoruz. Boş veritabanı veya `k <= 0` ise
  hiç sorgu atmadan `[]` dönüyoruz.
- Soru `embed_query` ile vektöre çevrilir, `collection.query(...)` çağrılır.
- `doc_id` verilirse `where={"doc_id": doc_id}` filtresi eklenir: sadece o belgede ara.
- Chroma bir **mesafe** (distance) döner. Kosinüs uzayında
  `distance = 1 - cosine_similarity`, dolayısıyla `score = 1 - distance` bize kosinüs
  benzerliğini geri verir. Skor 1'e ne kadar yakınsa chunk o kadar alakalı.

### Okuma ve listeleme

- `get_chunks` sayfa filtresi için Chroma'nın sorgu dilini kullanır:
  `{"$and": [{"doc_id": doc_id}, {"page": {"$in": pages}}]}`. Sonuç `chunk_idx`'e göre
  sıralanır, çünkü Chroma sıralama garantisi vermez.
- `list_documents` tek bir `get(include=["metadatas"])` çağrısıyla bütün metadata'yı
  çeker, `doc_id`'ye göre gruplar: chunk sayısı = o belgeye ait kayıt sayısı,
  sayfa sayısı = `doc_pages` metadata'sı.

## Önemli kavramlar

- **Vektör veritabanı**: vektörleri ve yanlarındaki metadata'yı saklayıp "bu vektöre
  en yakın olanlar hangileri?" sorusunu hızlı cevaplayan veritabanı.
- **Koleksiyon (collection)**: Chroma'daki tablo benzeri yapı. Biz tek bir `"chunks"`
  koleksiyonu kullanıyoruz; belgeleri `doc_id` metadata'sıyla ayırıyoruz.
- **Metadata filtresi (`where`)**: vektör aramasını belirli kayıtlarla sınırlamak.
  Örnek: sadece seçilen PDF'te arama yapmak.
- **Kosinüs mesafesi / benzerliği**: iki vektör arasındaki açıya bakar, uzunluğa
  bakmaz. `hnsw:space = "cosine"` bu yüzden seçildi; embedder'larımız zaten
  L2-normalize vektör üretiyor.
- **HNSW (Hierarchical Navigable Small World)**: yaklaşık en yakın komşu (ANN)
  arama algoritması. Çok hızlıdır ama sonuç %100 kesin olmayabilir.
- **Kalıcılık (persistence)**: `PersistentClient` veriyi diske yazar; yeni bir
  `VectorStore` aynı klasörü açınca eski belgeleri görür.
- **Idempotent ekleme**: aynı belgeyi tekrar eklemek sonucu değiştirmez (önce silip
  sonra ekliyoruz), `test_add_document_twice_replaces` bunu doğrular.
- **Test izolasyonu**: her test kendi `tmp_path` klasörünü kullanır; testler birbirinin
  verisini görmez.

## Mülakatta sorulabilecek sorular

1. **Vektör veritabanı ile klasik SQL veritabanı arasındaki fark nedir?**
   SQL tam eşleşme ve aralık sorgularında iyidir; vektör veritabanı "anlamca en yakın"
   kayıtları bulmak için en yakın komşu araması yapar.
2. **Chroma neden "distance" döndürüyor, siz neden `1 - distance` kullandınız?**
   Kosinüs uzayında mesafe `1 - benzerlik`tir. Skoru benzerlik olarak tutmak,
   `min_score` eşiğini (Task 7) daha anlaşılır kılar.
3. **Aynı PDF iki kez yüklenirse ne olur?**
   `add_document` önce eski kayıtları siler, sonra yenilerini ekler; chunk sayısı
   ikiye katlanmaz.
4. **Koleksiyonda 5 kayıt varken `k=10` istenirse ne olur?**
   `search` `n_results`'ı `count()` ile sınırlar ve 5 sonuç döner; boşsa `[]` döner.
5. **HNSW kesin sonuç verir mi?**
   Hayır, yaklaşık sonuç verir; hız için küçük bir doğruluk kaybını kabul eder. Küçük
   koleksiyonlarda pratikte fark edilmez.
6. **Neden embedding'i Chroma'ya bırakmadınız?**
   Hangi modelin kullanıldığını biz kontrol etmek istiyoruz (e5 önekleri, normalize
   etme) ve testlerde modeli tamamen sahte bir `HashEmbedder` ile değiştirebilmek için.
7. **Tek koleksiyon mu, belge başına koleksiyon mu?**
   Tek koleksiyon + `doc_id` filtresi: hem tüm notlarda hem tek belgede arama aynı
   kodla yapılır; belge silmek de `delete(where=...)` ile tek çağrıdır.
