# 06 – Yerel LLM: Ollama ile yapılandırılmış (JSON) cevaplar

Bu not, RAG hattının "beyin" katmanını anlatıyor: kendi bilgisayarımızda çalışan bir dil
modeline (`gemma2:9b`) soru sormak ve cevabı serbest metin yerine **şemaya uyan JSON**
olarak almak. Hatalar da öğrenciye anlaşılır bir mesajla iletiliyor.

---

## Ne yaptık?

- **`src/study_assistant/llm.py`** yazdık. İçinde:
  - Üç hata sınıfı: `LLMError(RuntimeError)` (genel hata), onun alt sınıfları
    `LLMUnavailableError` (Ollama çalışmıyor) ve `ModelNotFoundError` (model indirilmemiş).
  - `LLMClient` adında bir `Protocol`: uygulamanın LLM'den beklediği iki metot.
    - `complete_json(system, user, schema)`: Pydantic şemasına uyan bir nesne döner.
    - `check()`: model kullanıma hazır değilse hata fırlatır.
  - `OllamaClient(host, model, timeout)`: bu protokolün gerçek uygulaması.
    `ollama.Client` nesnesini `__init__` içinde **bir kez** oluşturur.
- **`tests/test_llm.py`**: 15 hızlı test + 1 `live` test. Hızlı testler
  `monkeypatch.setattr(ollama, "Client", StubClient)` ile gerçek istemciyi sahte bir
  sınıfla değiştirir; ağa hiç çıkılmaz. `test_live_ollama_roundtrip` ise gerçek
  `gemma2:9b` modeline gider ve varsayılan `pytest` çalıştırmasında atlanır.
- `tests/fakes.py` içindeki `FakeLLM` zaten aynı şekle (`complete_json` + `check`)
  sahipti; sonraki görevlerde (cevaplama, quiz, notlama) testler onu kullanacak.

## Neden?

- **Ücretsiz ve gizli.** Proje kuralı "ücretli API yok, API anahtarı yok". Ollama modeli
  yerelde çalıştırır; ders notların bilgisayarından çıkmaz.
- **Serbest metni kod okuyamaz.** "Puan: seksen civarı, gayet iyi" gibi bir cevabı
  ayrıştırmak kırılgandır. Modelden `{"score": 80, "feedback": "iyi"}` gibi JSON isteyip
  Pydantic ile doğrularsak, geri kalan kod tipli bir `LLMGrade` nesnesiyle çalışır.
- **Tekrarlanabilirlik.** `temperature=0` ile aynı soruya (neredeyse) hep aynı cevap
  gelir. Notlama ve testler için önemli.
- **Hata mesajı bir çözüm söylemeli.** Öğrenci stack trace görmemeli; "Ollama çalışmıyor,
  `ollama serve` ile başlat" veya "`ollama pull gemma2:9b` çalıştır" görmeli. API
  katmanı (Görev 10) bu hata sınıflarını 503 cevabına çevirecek.
- **Protocol ile bağımlılığı gevşetmek.** Diğer modüller `OllamaClient`'a değil
  `LLMClient` arayüzüne bağlı. Testlerde `FakeLLM`, gerçekte `OllamaClient` verilir.

## Nasıl çalışıyor?

`complete_json` akışı:

1. Mesaj listesi kurulur: `{"role": "system", ...}` (talimatlar) ve
   `{"role": "user", ...}` (asıl soru).
2. `self._client.chat(...)` çağrılır:
   - `format=schema.model_json_schema()`: Pydantic modelinden üretilen JSON Schema.
     Ollama bunu **yapılandırılmış çıktı** için kullanır; model üretim sırasında bu
     şemaya uyan token'larla sınırlandırılır.
   - `options={"temperature": 0, "num_ctx": 8192}`: rastgelelik kapalı, bağlam
     penceresi 8192 token (Ollama'nın varsayılanı daha küçük olabiliyor; RAG
     prompt'larımız kesilmesin diye açıkça veriyoruz).
3. Cevap metni `response.message.content` içindedir. `schema.model_validate_json(...)`
   ile doğrulanır.
4. Doğrulama başarısız olursa (`ValidationError` / `JSONDecodeError`) **bir kez** yeniden
   denenir: önceki mesajlara modelin hatalı cevabı (`assistant`) ve şu kullanıcı mesajı
   eklenir: "Your previous reply was not valid JSON for the schema. Reply again with
   JSON only." Yeni bir liste kurulur; eski liste değiştirilmez.
5. İkinci deneme de başarısızsa `LLMError` fırlatılır. Sonsuz döngü yok.

Hata çevirisi (`_call` metodu, hem `chat` hem `check` için ortak):

| Kütüphanenin fırlattığı                         | Bizim fırlattığımız                       |
|-------------------------------------------------|-------------------------------------------|
| `ConnectionError` / `httpx.ConnectError`         | `LLMUnavailableError` (host + `ollama serve`) |
| `httpx.TimeoutException`                         | `LLMError` ("took too long ... timeout 180s") |
| `ollama.ResponseError`, `status_code == 404`     | `ModelNotFoundError` (`ollama pull <model>`) |
| Diğer `ollama.ResponseError`                     | `LLMError` (sunucunun mesajıyla)           |

Not: `ollama` 0.6.3 istemcisi sunucuya bağlanamayınca `httpx.ConnectError`'u yakalayıp
Python'un yerleşik `ConnectionError`'ını fırlatıyor (kaynak kodda `_request_raw`).
Bunu `http://localhost:9` gibi boş bir porta bağlanarak elle doğruladık. Yine de
güvenli tarafta kalmak için ikisini de yakalıyoruz.

`check()` ise `self._client.show(model)` çağırır: model kurulu değilse Ollama 404 döner,
o da `ModelNotFoundError` olur. Uygulama açılırken bu kontrolle erken uyarı verilebilir.

Canlı test sonucu (bu makinede, model önceden yüklenmişken): yaklaşık **9.8 saniye**,
cevap `LLMGrade(score=100, feedback='Correct!')`. İlk yüklemede model belleğe alınırken
süre daha uzun olabilir; bu yüzden `llm_timeout_s` varsayılanı 180 saniye.

## Önemli kavramlar

- **Yerel çıkarım (local inference):** Modelin ağırlıkları senin diskinde, hesaplama
  senin CPU/GPU'nda yapılır. Ollama bir HTTP sunucusu (`localhost:11434`) olarak çalışır;
  Python istemcisi ona istek atar.
- **Kuantizasyon:** `gemma2:9b` gibi modeller genellikle 4-bit'e sıkıştırılmış olarak
  gelir; böylece 9 milyar parametre birkaç GB belleğe sığar.
- **System / user / assistant rolleri:** `system` modelin nasıl davranacağını, `user`
  soruyu, `assistant` modelin önceki cevaplarını taşır. Yeniden denemede hatalı cevabı
  `assistant` olarak geri vermek, modele neyi düzeltmesi gerektiğini gösterir.
- **Yapılandırılmış çıktı (structured output):** JSON Schema verilerek modelin çıktısı
  biçimsel olarak kısıtlanır. Yine de **içerik** yanlış olabilir (ör. var olmayan kaynak
  numarası); bu yüzden Pydantic doğrulaması ve sonraki görevlerdeki ek kontroller şart.
- **Temperature:** Olasılık dağılımının "sıcaklığı". 0 = her adımda en olası token
  (deterministik'e yakın), yüksek değer = daha yaratıcı ama daha tutarsız.
- **Bağlam penceresi (`num_ctx`):** Modelin aynı anda görebildiği token sayısı. Prompt +
  cevap bunu aşarsa baştaki metin kesilir; RAG'de bu, kaynakların kaybolması demek.
- **Protocol (yapısal tipleme):** `LLMClient`'tan miras almaya gerek yok; aynı
  metotlara sahip her sınıf (`OllamaClient`, `FakeLLM`) uyumludur.
- **Monkeypatch ile test:** `ollama.Client` testte `StubClient` ile değiştirilir.
  Sahte istemci cevapları sıradan döner ve her çağrıyı kaydeder; böylece gönderilen
  `format`, `options` ve mesajları doğrulayabiliriz.
- **Hata zinciri (`raise ... from exc`):** Kullanıcıya sade mesaj gösterirken orijinal
  hatayı `__cause__` olarak saklar; hata ayıklarken kaybolmaz.

## Mülakatta sorulabilecek sorular

1. **LLM'den neden serbest metin yerine JSON istiyorsunuz?**
   Kod tarafında güvenilir ayrıştırma için. Şema ile çıktı biçimi kısıtlanır, Pydantic
   ile tip ve alanlar doğrulanır; hatalı cevap sessizce sisteme girmez.
2. **JSON Schema verdiniz, yine de neden doğrulama ve retry var?**
   Kısıtlı üretim biçimi büyük ölçüde garanti eder ama model yarıda kesilebilir
   (bağlam/token sınırı) veya sunucu şemayı tam uygulamayabilir. Savunmacı tasarım: bir
   kez yeniden dene, sonra net bir `LLMError` ver.
3. **Neden sadece bir kez yeniden deniyorsunuz?**
   Yerel modelde her çağrı saniyeler sürer. İki başarısızlık genelde prompt'ta bir sorun
   olduğunu gösterir; daha fazla deneme kullanıcıyı bekletir.
4. **`temperature=0` ne sağlar, dezavantajı ne?**
   Tekrarlanabilir, tutarlı cevaplar (notlama için adil). Dezavantaj: çeşitlilik azalır;
   quiz soruları birbirine benzeyebilir.
5. **Ollama kapalıysa kullanıcı ne görür?**
   `LLMUnavailableError`: "Ollama is not running at http://localhost:11434. Start it
   with: ollama serve". API bunu 503'e, arayüz bir uyarı bandına çevirir.
6. **Testler gerçek modele gitmeden nasıl çalışıyor?**
   `ollama.Client` monkeypatch ile `StubClient`'a çevriliyor. Gerçek model testi
   `@pytest.mark.live` ile işaretli ve varsayılan çalıştırmada hariç tutuluyor.
7. **`Protocol` ile soyut temel sınıf (ABC) arasındaki fark nedir?**
   ABC'de miras zorunludur (nominal tipleme). `Protocol`'de yalnızca metot imzaları
   eşleşmeli (yapısal tipleme); `FakeLLM` hiçbir şeyden miras almadan uyumlu.
8. **Bulut LLM yerine yerel LLM'in artı ve eksileri?**
   Artı: ücretsiz, gizlilik, internet gerekmez. Eksi: daha küçük model = daha düşük
   kalite, donanıma bağlı hız, ilk yüklemede gecikme.
