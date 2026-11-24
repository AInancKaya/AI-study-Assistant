# 08 – Quiz üretimi, doğrulama ve SQLite'ta saklama

Bu not, yüklenen bir PDF'ten (ders notundan) yerel LLM ile **çoktan seçmeli** ve
**açık uçlu** sorular üreten, modelin çıktısını sıkıca doğrulayan ve quiz'i diske kaydeden
kısmı anlatıyor.

---

## Ne yaptık?

- **`src/study_assistant/quiz.py`** yazdık:
  - `select_quiz_chunks(chunks, max_chunks)`: dokümanın tamamına **eşit aralıklarla**
    yayılmış en fazla `max_chunks` parça seçer (ilk ve son parça her zaman dahil).
  - `fit_to_budget(chunks, budget=MAX_SOURCE_CHARS)`: seçilen parçaların toplam metni
    12000 karakteri aşarsa her parçayı `12000 // parça_sayısı` karaktere kısaltır
    (son boşluktan keser, sonuna `…` ekler).
  - `build_user_prompt(sources, n_mc, n_open)`: `format_sources(...)` çıktısı ve ardından
    `Create exactly 3 multiple_choice and 2 open_ended questions.` satırı. Sayısı 0 olan
    tür hiç yazılmaz.
  - `generate_quiz(doc_id, store, llm, repo, *, n_multiple_choice=3, n_open_ended=2,
    max_chunks)`: asıl fonksiyon. Sayıları kontrol eder, LLM'i `QUIZ_SYSTEM` ve
    `LLMQuiz` şemasıyla çağırır, soruları doğrular, quiz'i kaydeder.
  - `QuizGenerationError`: model **hiç geçerli soru** üretemezse fırlatılır.
- **`src/study_assistant/quiz_store.py`** yazdık:
  - `QuizRepository(db_path)`: `quizzes(id, doc_id, created_at, data)` tablosunu yönetir;
    `save`, `get`, `list(doc_id=None)` (en yeni önce) metodları var.
- **`tests/test_quiz.py`**: parça seçimi, geçerli quiz, geçersiz soruların atılması,
  kaynak→sayfa eşlemesi, hatalı sayılar, bilinmeyen doküman, bağlam bütçesi ve
  veritabanı gidiş-dönüşü için testler. LLM yerine `FakeLLM`, embedding yerine
  `HashEmbedder` kullanılıyor; ağa hiç gidilmiyor.

## Neden?

- **Bağlam penceresi küçük.** `gemma2:9b` yaklaşık 8k token görebiliyor. 300 sayfalık bir
  notun tamamını gönderemeyiz; bu yüzden en fazla `quiz_max_chunks` (varsayılan 10) parça
  seçiyoruz ve toplam metni `MAX_SOURCE_CHARS = 12000` karakterle sınırlıyoruz
  (Türkçe metinde kabaca 3–4 bin token). Geri kalan yer sistem mesajına ve modelin
  cevabına kalıyor.
- **Eşit dağılım = konu kapsamı.** Sadece ilk 10 parçayı alsaydık quiz hep ilk ünite
  hakkında olurdu. Eşit aralıklı seçim, baştan sona her bölümden örnek alır.
- **Küçük modeller hata yapar.** 3 şıklı soru, `correct_option = 7`, aynı şıkkın iki kez
  yazılması, boş cevap anahtarı… Bunları kullanıcıya göstermek yerine **koda** yakalatıyoruz.
  Kötü soruyu atmak, tüm quiz'i 500 hatasıyla çöpe atmaktan iyidir.
- **Quiz'in kalıcı olması lazım.** Öğrenci quiz'i çözdükten sonra (Görev 9'da) cevaplar
  notlanacak. Doğru şık ve referans cevap sunucuda saklanmalı, öğrenciye gitmemeli
  (`Quiz.public()` bunları siler). Bu yüzden quiz'i SQLite'a yazıyoruz.

## Nasıl çalışıyor?

1. **Sayı kontrolü.** Her tür 0..10, toplam 1..10 olmalı. `(0, 0)` veya `(8, 5)` →
   `ValueError`. Bilinmeyen doküman (`store.has_document` false) → `KeyError(doc_id)`.
   İkisi de LLM çağrılmadan olur.
2. **Parça seçimi.** `store.get_chunks(doc_id)` parçaları `chunk_idx` sırasıyla verir.
   `n` parça içinden `m = min(max_chunks, n)` tane seçilir; indeksler
   `round(i * (n-1) / (m-1))`. Örnek: 100 parça, 10 seçim → 0, 11, 22, …, 88, 99.
3. **Bütçe.** Toplam karakter > 12000 ise her parça `12000 // m` karaktere kesilir.
   10 parça × 1500 karakter = 15000 → her biri en fazla 1200 karakter. Kelimenin ortasından
   kesmemek için son boşluğa geri gidilir ve `…` eklenir. Hiç parça kaybolmaz; [1]..[10]
   hepsi prompt'ta kalır.
4. **Prompt.** Kullanıcı mesajı, `rag.py`'deki gibi numaralı kaynak blokları ve tek bir
   talimat satırıdır. Kurallar (`QUIZ_SYSTEM`) sistem mesajında.
5. **LLM çağrısı.** `llm.complete_json(QUIZ_SYSTEM, user, LLMQuiz)`. JSON şeması Ollama'ya
   gönderilir; bozuk JSON gelirse `LLMClient` bir kez yeniden dener (Görev 6).
6. **Doğrulama** (`_validate`):
   - Soru metni kırpılır (`strip`), boşsa atılır.
   - **Çoktan seçmeli:** tam 4 şık, hepsi kırpılmış ve boş değil, büyük/küçük harf
     duyarsız tekrar yok (`casefold`), `correct_option` 0..3. Referans cevap boşsa doğru
     şıkkın metni kullanılır.
   - **Açık uçlu:** `options` `None` ya da `[]` olmalı, referans cevap boş olmamalı.
     Kayıtta `options=None`, `correct_option=None`.
7. **Kırpma.** İstenenden fazla soru gelirse her türden ilk N geçerli soru tutulur
   (`_valid_questions`). Sıra modelin sırasıdır; id'ler `q1..qN` olarak sonradan verilir.
8. **Kaynak → sayfa.** `source_ids` 1'den başlar; aralık dışındakiler yok sayılır.
   Geçerli olanların sayfaları sıralı ve tekrarsız yazılır. Hiç geçerli id yoksa, seçilen
   parçaların tüm sayfaları kullanılır (en azından "bu dokümanın bu sayfalarından" bilgisi).
9. **Sonuç.** Hiç geçerli soru yoksa `QuizGenerationError`. Varsa
   `Quiz(id=uuid4().hex[:12], ...)` oluşturulur, `repo.save(quiz)` ile kaydedilir.

### `QuizRepository`

```sql
CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY, doc_id TEXT, created_at TEXT, data TEXT
)
```

- `data` sütununda quiz'in tamamı JSON olarak durur (`quiz.model_dump_json()`); okurken
  `Quiz.model_validate_json(...)` ile geri Pydantic modeline döner.
- `created_at` ISO 8601 metnidir (`2026-11-24T12:00:00+00:00`). Hep UTC yazdığımız için
  metin sıralaması zaman sıralamasıyla aynıdır → `ORDER BY created_at DESC`.
- Her işlemde yeni bağlantı açılır ve `with closing(sqlite3.connect(...))` ile kapanır.
  FastAPI istekleri farklı thread'lerde çalışır; tek bir paylaşılan bağlantı sorun çıkarır.
- Klasör yoksa (`data/`) `mkdir(parents=True)` ile oluşturulur.

## Önemli kavramlar

- **Yapılandırılmış çıktı (structured output):** Modelden serbest metin değil, şemaya
  uyan JSON istemek. Şema şekli garanti eder, **anlamı** garanti etmez; o yüzden
  aralık kontrollerini kodda yapıyoruz.
- **Savunmacı doğrulama (defensive validation):** LLM çıktısına kullanıcı girdisi gibi
  davranmak: güvenme, kontrol et, geçersizi at.
- **Bağlam bütçesi (context budget):** Prompt'a ne kadar metin sığacağını önceden
  hesaplamak. Aşılırsa model baştaki metni "unutur" ya da hata verir.
- **Eşit aralıklı örnekleme:** Uzun bir diziden temsil edici küçük bir alt küme seçmek.
- **Grounding (dayanak):** Her soru `source_pages` ile notlardaki sayfalara bağlanır;
  öğrenci "bu soru nereden çıktı?" diye bakabilir.
- **Repository deseni:** Veritabanı detaylarını (`sqlite3`, SQL) tek bir sınıfın içine
  saklamak. Diğer kod sadece `save/get/list` bilir.
- **Doküman olarak JSON saklama:** İlişkisel tabloya her alanı ayrı sütun yapmak yerine
  modeli tek JSON metni olarak saklamak; sorgulanan alanlar (`doc_id`, `created_at`)
  ayrıca sütun olur.

## Mülakatta sorulabilecek sorular

1. **LLM yanlış formatta soru üretirse ne yaparsınız?**
   Şema ile JSON şeklini zorlarız, bozuk JSON'da bir kez yeniden deneriz; sonra her soruyu
   kodda doğrularız (4 şık, geçerli doğru şık indeksi, boş olmayan cevap). Geçersizleri
   atar, hiç geçerli kalmazsa anlamlı bir hata (`QuizGenerationError`) döneriz.
2. **Uzun bir dokümandan neden sadece birkaç parça gönderiyorsunuz?**
   Model penceresi 8k token. Eşit aralıklı seçim + karakter bütçesi ile hem pencereye
   sığarız hem de dokümanın her bölümünü kapsarız.
3. **Neden kırpmada son boşluğa kadar geri gidiyorsunuz?**
   Yarım kelime modeli şaşırtabilir; özellikle Türkçede ekler anlamı değiştirir. `…`
   işareti metnin devam ettiğini belirtir.
4. **SQLite bağlantısını neden her işlemde açıp kapatıyorsunuz?**
   `sqlite3` bağlantısı varsayılan olarak thread'ler arası paylaşılamaz. Yerel, küçük bir
   uygulamada bağlantı açmak ucuzdur; böylece thread güvenliği derdi kalmaz.
5. **`created_at`'i metin olarak saklamak sıralamayı bozmaz mı?**
   Hepsi aynı saat diliminde (UTC) ISO 8601 ise bozmaz; metin sırası zaman sırasıyla aynıdır.
6. **Doğru cevabı öğrenciden nasıl saklıyorsunuz?**
   Tam quiz sunucuda SQLite'ta durur; API'ye `Quiz.public()` döner, o da `correct_option`
   ve `reference_answer` alanlarını çıkarır.
7. **Modelin uydurduğu kaynak numaralarıyla ne yapıyorsunuz?**
   1..N dışındakileri yok sayarız; hiç geçerli yoksa seçilen parçaların tüm sayfalarını
   veririz ki soru yine de notlara bağlı kalsın.
