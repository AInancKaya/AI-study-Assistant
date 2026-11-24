# 02 – PDF Okuma (metin katmanı, Türkçe güvenli normalizasyon)

Bu not, ders notlarından gelen PDF dosyalarını sayfa sayfa metne çeviren `ingest`
katmanını anlatıyor. Bu katman olmadan RAG'in arayacağı bir şey de olmaz.

---

## Ne yaptık?

- **`src/study_assistant/ingest/pdf.py`** yazdık. Ana giriş noktası
  `extract_pages(data, filename)`: PDF baytlarını alır, her sayfanın metnini
  `Page` modeli olarak döndürür (1 tabanlı sayfa numarası).
- **Hata sınıfları:** `InvalidPDFError` (dosya okunamayan bir PDF) ve
  `ScannedPDFError` (metin katmanı olmayan, yani taranmış PDF). İkisi de
  `ValueError` alt sınıfı; API katmanı bunları 400 / 422 cevaplarına çevirecek.
- **`make_doc_id`** içeriğin SHA-256 özetinin ilk 12 karakterini döndürür. Aynı dosya
  her zaman aynı kimliği alır.
- **`count_pages`** sayfa sayısını verir; yükleme sırasında 300 sayfa sınırını
  kontrol etmek için kullanılacak.
- **Test fikstürleri:** `scripts/make_pdf.py` bir `.md` kaynağından reportlab ile PDF
  üretir. `tests/fixtures/turkce_notlar.pdf` (3 sayfa, Türkçe karakterler ve satır sonu
  tire kırılması içerir) ve `empty.pdf` (boş 2 sayfa) bu script ile üretildi ve
  repoya eklendi.

## Neden?

- **Metin katmanı ile taranmış PDF farkı.** Ders notu PDF'lerinin çoğu bir Word
  dosyasından "metin olarak" çıkarılır; içinde gerçek karakterler vardır. Taranmış bir
  PDF ise sadece resimdir. Resmi okumak OCR gerektirir; bu proje OCR yapmıyor, bu
  yüzden metin yoksa kullanıcıya açık bir hata veriyoruz (`ScannedPDFError`).
- **Neden `pypdf`, PyMuPDF değil?** PyMuPDF hızlı ve güçlü, ama AGPL lisanslı. Bu proje
  MIT lisanslı ve "AGPL bağımlılık yok" kuralına sahip. `pypdf` saf Python, BSD
  lisanslı ve bu iş için yeterli.
- **NFC normalizasyonu.** Türkçe "ğ", "ş", "ı" gibi harfler Unicode'da iki farklı
  biçimde yazılabilir: tek kod noktası (`ş` = U+015F) ya da "s" + birleşen cedilla
  (U+0327). Görünüşleri aynı olsa da Python'da `==` ve arama eşleşmeleri farklı
  sonuç verir. NFC her zaman tek kod noktası biçimine çevirir.
- **Satır sonu tire birleştirmesi.** PDF'te "öğrenme-" kelimesi satır sonunda
  bölünmüş olabilir. Birleştirmezsek RAG "öğrenme" ve "nin" gibi parçalar arar ve
  kelimeyi bulamaz.

## Nasıl çalışıyor?

Akış şöyle:

1. `_reader(data)` bayt dizisini `io.BytesIO` ile `PdfReader`'a verir. Bozuk veya PDF
   olmayan veri burada `InvalidPDFError` olur.
2. `_page_texts(data)` her sayfa için `page.extract_text()` çağırır. Boş dönen
   sayfalar `""` olur.
3. `normalize_text(raw)` her sayfa metnini şu sırayla temizler:
   - `unicodedata.normalize("NFC", ...)`
   - `_HYPHEN_BREAK` regex'i: `(?<=\w)-[ \t]*\n[ \t]*(?=\w)` deseni, satır sonundaki
     tireyi ve sonraki satırın başındaki boşlukları siler. Önündeki ve arkasındaki
     karakter harf olmak zorunda, bu yüzden "- " gibi liste işaretleri etkilenmez.
   - `_SPACE_RUN` ile ardışık boşluklar tek boşluğa iner.
   - Boş satırlar atılır.
4. Metni boş olan sayfalar listeye girmez (`if (text := ...)`).
5. Sayfa başına boşluk olmayan karakter sayısı hesaplanır. Hiçbir sayfa en az
   `MIN_TEXT_CHARS` (20) karaktere sahip değilse `ScannedPDFError` atılır. Kontrol
   sayfa bazında yapılır: toplamda 20'yi geçen ama her sayfası kısa olan (sadece
   "Sayfa 1", "Sayfa 2" gibi altbilgi) bir tarama da reddedilir.

Testler `tests/test_pdf.py` içinde: Türkçe metin korunuyor mu, tire birleşiyor mu,
kimlik kararlı mı, taranmış ve bozuk dosyalar doğru hatayı veriyor mu, sayfa sayısı
doğru mu.

## Önemli kavramlar

- **Metin katmanı (text layer):** PDF içindeki, kopyalanabilir ve aranabilir karakter
  dizisi. Yoksa dosya bir resimdir.
- **Unicode normalizasyonu (NFC / NFD):** Aynı görünen metnin farklı kod noktası
  dizileriyle temsil edilmesini önler. Arama ve karşılaştırma için şart.
- **Satır sonu tire (hyphenation):** Dizgi motorunun uzun kelimeyi satır sonunda
  bölmesi. Anlamı korumak için geri birleştirilmeli.
- **Regex lookbehind / lookahead:** `(?<=\w)` ve `(?=\w)` karakteri eşleşmeye dahil
  etmeden çevresine bakar; yani sadece tire silinir, harfler kalır.
- **Fikstür (fixture):** Testlerin tekrarlanabilir girdisi. Üretilen PDF'i repoya
  koymak, testin her makinede aynı dosyayla çalışmasını sağlar.
- **Lisans uyumu:** AGPL bir kütüphane, MIT projesini kullanım şekline göre
  lisanslama yükümlülüğü getirebilir. Bu yüzden bağımlılık seçimi lisansla birlikte
  yapılır.

## Mülakatta sorulabilecek sorular

1. Bir PDF'in metin olup olmadığını nasıl anlarsın? Taranmış PDF'i neden sessizce
   geçmek yerine hata olarak bildirirsin?
2. Unicode NFC ve NFD farkı nedir? Türkçe metinde arama yaparken neden önemlidir?
3. Satır sonu tire birleştirmesini neden regex ile yaptın? Hangi yanlış pozitifleri
   doğurabilir (ör. gerçek bir "iki-\nyönlü" ifadesi)?
4. PyMuPDF yerine pypdf seçmenin bedeli nedir? Performans ve özellik açısından ne
   kaybedersin?
5. `make_doc_id` dosya adı yerine içerik hash'i kullanıyor. Bunun sebebi ve
   sınırı nedir?
6. Testlerde PDF'i neden kod içinde üretmek yerine repoya ikili dosya olarak
   ekledik? Bu kararın avantaj ve dezavantajları nelerdir?
