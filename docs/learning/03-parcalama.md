# 03 – Parçalama (sayfa farkında, örtüşmeli chunk'lar)

Bu not, `ingest` katmanının ikinci adımını anlatıyor: sayfa metinlerini embedding ve
arama için uygun parçalara (chunk) bölmek. Bu parçalar, cevaplardaki sayfa atıflarının
da temelidir.

---

## Ne yaptık?

- **`src/study_assistant/ingest/chunking.py`** yazdık. Ana giriş noktası
  `chunk_pages(pages, size=1500, overlap=200)`: `Page` listesini alır, `Chunk` listesi
  döndürür.
- **Sınır kuralı:** Her chunk tek bir sayfadan gelir. İki sayfa asla birleşmez, bu
  yüzden her chunk'ın `page` alanı kesin bir atıf verir.
- **Bölme sırası:** Önce paragraflar (boş satır), sonra cümleler (`.`, `!`, `?`, `…`
  sonrasındaki boşluk), en son da tek başına `size` karakterden uzun bir cümle için sabit
  uzunlukta pencereler (`_hard_split`).
- **Örtüşme:** Yeni bir chunk, öncekinin sonundan en az `overlap` karakter taşır. Mümkünse
  bu tam cümlelerdir; değilse öncekinin son `overlap` karakteri kullanılır.
- **Kimlik:** `chunk_idx` tüm liste boyunca 0'dan artar; `id` `"{doc_id}:{page}:{chunk_idx}"`
  biçimindedir.
- **Doğrulama:** `size > 0` ve `0 <= overlap < size` değilse `ValueError` fırlatılır.
- **Testler:** `tests/test_chunking.py` 13 test içerir (kısa sayfa, boyut sınırı, sayfa
  sınırı, kimlikler, örtüşme, uzun cümle, Türkçe karakterler, boş girdi, boşluk sayfası,
  geçersiz parametre).

## Neden?

**Neden parçalıyoruz?** Bir ders notu PDF'i onlarca sayfa, on binlerce karakter olabilir.
Modelin bağlam penceresi (bizde `gemma2:9b` için 8k token bütçesi) bunların hepsini
kaldıramaz ve arama tüm sayfayı tek bir vektöre sıkıştırırsa ilgili cümle kaybolur.
Küçük parçalar üzerinde arama yaparsak, sorunun cevabını içeren kısmı getirebiliriz.

**Boyut (`size`) ödünleşimi:**
- Küçük chunk: arama daha kesin olur, ama bir fikir iki parçaya bölünürse cevap eksik
  kalabilir ve embedding modelinin bağlamı zayıflar.
- Büyük chunk: daha fazla bağlam taşır, ama alakasız cümleler de vektöre karışır ve
  LLM'in prompt'u şişer.
- 1500 karakter, ders notu paragrafının kabaca birkaç katına denk gelir ve 8k token
  bütçesine sığar.

**Örtüşme (`overlap`) ödünleşimi:** Bir cümle iki chunk'ın sınırına denk gelirse, sınırın
iki yanındaki bağlam ayrı ayrı eksik kalır. 200 karakterlik örtüşme bu kesikleri azaltır.
Bedeli: depolama ve embedding maliyeti biraz artar, aynı metin iki kez gelir.
`overlap = 0` hiç örtüşme olmadığı anlamına gelir: chunk'lar birbirinin metnini tekrar etmez.

**Sayfa metadatası neden önemli?** Kullanıcı "cevap nerede?" diye sorduğunda, "3. sayfa,
notlar.pdf" gibi bir atıf göstermek istiyoruz. Chunk'ı üreten sayfa numarası ve dosya
adı chunk'ın üzerinde taşındığı için, arama sonucu doğrudan atıfa dönüşebilir. Bir
chunk iki sayfayı birleştirseydi bu atıf yanlış olurdu.

## Nasıl çalışıyor?

Algoritma üç aşamada ilerler.

**1. Parçalara ayırma (`_pieces`).** Sayfa metni `\n\s*\n` ile paragraflara bölünür. Her
paragraftaki kelimeler tek boşlukla birleştirilir (satır sonları dahil), sonra
`(?<=[.!?…])\s+` ile cümlelere ayrılır. Cümle `size`'dan uzunsa `_hard_split` onu
`size - overlap` uzunluğunda pencerelere böler. Her parçanın bir `sep` alanı vardır: sayfa
metninde kendisinden önce gelen ayraç (`""`, `" "` veya `"\n\n"`). Pencereler arasında
ayraç yoktur, çünkü bunlar aynı cümlenin devamıdır.

**2. Paketleme (`_pack`).** Parçalar sırayla bir chunk'a eklenir. Yeni parça sığmazsa
mevcut chunk kapatılır ve `_seed` ile bir sonraki chunk'a taşınacak örtüşme hesaplanır.

**3. Örtüşme seçimi (`_seed`).** Önce `_trailing_sentences` dener: kapanan chunk'ın
sonundan tam cümleleri, toplam uzunluğu `overlap`'a ulaşana kadar alır. Bu küme yeni
parçanın yanına sığmazsa (boyut sınırı her şeyden önce gelir) veya hiç ulaşamazsa, yedek
yöntem devreye girer: kapanan chunk'ın son `min(overlap, boşluk)` karakteri alınır.

Örnek (size=500, overlap=80, her cümle ~95 karakter):

```
chunk 1: [S1 S2 S3 S4 ........ S5]          <- 500 karaktere yakın
chunk 2:       [S5 S6 S7 .......]           <- S5 taşındı: en az 80 karakter örtüşme
```

Sağlamlık için kritik nokta: her chunk, sayfa metninin **bitişik bir parçası** olarak
kurulur. Taşınan örtüşme her zaman kapanan chunk'ın sonudur ve yeni parçanın önünde gelir.
Bu yüzden `chunk.text`, normalize edilmiş sayfa metninin bir alt dizgisidir. Testler bu
özelliği `test_chunks_never_cross_pages` ile doğrular. Ayrıca 3000 rastgele girdi üzerinde
boyut, alt dizgi ve `chunk_idx` kuralları da kontrol edildi (ihlal yok).

**Boyut sınırı her zaman kazanır.** Örtüşme sadece boşluk izin verdiği kadar eklenir. Yani
bazı durumlarda örtüşme `overlap`'tan kısa kalabilir; bu bilinçli bir tercihtir çünkü
embedding ve LLM girdisi için sert üst sınır önemlidir.

## Önemli kavramlar

- **Chunk:** Aramada ve embedding'de kullanılan, sayfa metninin bitişik bir parçası.
- **Örtüşme (overlap):** Ardışık iki chunk'ın paylaştığı metin. Sınırdaki bilgiyi korur.
- **Hiyerarşik bölme:** Paragraf → cümle → karakter sırasıyla, anlamı en çok koruyan
  sınırdan başlayıp gerekirse daha ince bir sınıra inmek.
- **Sabit pencere (hard split):** Doğal sınır yoksa karakter sayısına göre bölme. Cümleyi
  ortadan keser, bu yüzden yalnızca son çare olarak kullanılır.
- **Sayfa atıfı (citation):** Chunk'ın hangi sayfadan geldiği bilgisi. Cevapların güvenilir
  olmasını sağlar.
- **Token ve karakter:** Bütçe karakterle ölçülür (`size`), çünkü tokenizer modeline göre
  değişir. Karakter sayısı basit ve tahmin edilebilir bir üst sınırdır.

## Mülakatta sorulabilecek sorular

1. **Neden sabit boyutlu chunk yerine cümle bazlı bölme yapmadın?** Cümle bazlı bölme
   anlamı korur, ama uzunluğu öngörülemez. Biz hiyerarşik bir yaklaşım kullandık: önce
   doğal sınırlar, sonra sabit pencere. Böylece hem anlam hem boyut sınırı korunur.
2. **Örtüşmeyi neden tam cümleyle değil karakterle de yapıyorsun?** Tek bir cümle `size`'dan
   uzunsa tam cümle taşınamaz. Bu durumda son `overlap` karakter yedek olarak kullanılır.
   Bu, boyut sınırını bozmadan sınır bilgisini korur.
3. **Chunk'lar sayfa sınırını neden geçmez?** Atıf doğruluğu için. İki sayfayı birleştiren
   bir chunk "3. sayfa" diyemez. Ayrıca sayfa metni ayrı bir bağlamdır.
4. **`overlap` ne kadar olmalı?** Genellikle boyutun %10–20'si. Çok büyük örtüşme
   tekrarlı sonuçlar ve daha fazla maliyet demektir; çok küçük örtüşme sınır bilgisini
   kaybettirir. Değer `config.py`'de (`STUDY_CHUNK_OVERLAP`) ayarlanabilir ve Task 12'deki
   değerlendirmelerle doğrulanacak.
5. **Türkçe metinde dikkat edilmesi gereken bir şey var mı?** Bölme karakter bazlı
   olduğu için `ı`, `İ`, `ş`, `ğ` gibi karakterler bozulmaz; ancak Unicode normalizasyonu
   (NFC) `ingest/pdf.py`'de yapılmalı ki aynı harf farklı kod noktalarıyla gelmesin.
6. **`chunk_idx` neden liste genelinde artıyor, sayfa başına değil?** Her chunk'ın
   kimliği benzersiz olsun diye. `id` zaten sayfa numarasını içerdiği için sayfa içi
   sıralama da okunabilir kalır.
