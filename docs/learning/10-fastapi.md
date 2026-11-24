# 10 – FastAPI backend: çekirdek modülleri HTTP'ye açmak

Bu not, şimdiye kadar yazdığımız çekirdek modülleri (PDF okuma, parçalama, vektör
deposu, RAG, quiz, puanlama) tek bir **REST API** arkasında nasıl birleştirdiğimizi
anlatıyor. Streamlit arayüzü (sonraki görev) sadece bu API ile konuşacak.

---

## Ne yaptık?

- **`src/study_assistant/api/schemas.py`**: sadece HTTP katmanında var olan gövdeler.
  - `AskRequest(question, doc_id=None)`, `QuizRequest(doc_id, n_multiple_choice=3,
    n_open_ended=2)`, `GradeRequest(quiz_id, question_id, answer=None,
    selected_option=None)`, `HealthResponse(status, llm_ok, llm_detail, model)`.
  - Cevap modelleri (`DocumentInfo`, `Answer`, `QuizPublic`, `Grade`) yeniden
    yazılmadı; `study_assistant.models` içinden aynen kullanılıyor.
- **`src/study_assistant/api/main.py`**:
  - `create_app(settings=None, *, store=None, llm=None, quizzes=None)`: uygulamayı
    kuran fabrika fonksiyonu. Verilmeyen bağımlılıkları ayarlardan üretir:
    `VectorStore(data_dir/"chroma", E5Embedder(...))`, `OllamaClient(...)`,
    `QuizRepository(data_dir/"quizzes.db")`. Hepsi `Deps` adlı bir dataclass'a
    konup `app.state.deps` üzerinde saklanır.
  - Uç noktalar (endpoint'ler):

    | Yöntem ve yol | Başarılı cevap | Hatalar |
    |---|---|---|
    | `GET /health` | `HealthResponse` (her zaman 200) | — |
    | `POST /documents` | `DocumentInfo` 201, aynı dosya tekrar → 200 | 400, 413, 422 |
    | `GET /documents` | `list[DocumentInfo]` | — |
    | `DELETE /documents/{doc_id}` | 204 | 404 |
    | `POST /ask` | `Answer` | 400 boş soru, 404 bilinmeyen doküman |
    | `POST /quiz` | `QuizPublic` 201 | 400, 404, 502 |
    | `POST /grade` | `Grade` | 400, 404 |

  - Genel hata yakalayıcılar: `LLMUnavailableError` ve `ModelNotFoundError` → **503**,
    diğer `LLMError` → **502**; gövde her zaman `{"detail": "<mesaj>"}`.
- **`tests/test_api.py`**: `TestClient` ile 18 test. Gerçek Chroma (geçici klasörde)
  ama `HashEmbedder`; gerçek Ollama yerine `FakeLLM`. Ağa hiç çıkılmıyor.

## Neden?

- **Arayüz ile mantığı ayırmak.** Streamlit, ileride bir mobil uygulama ya da
  `curl` – hepsi aynı API'yi kullanabilir. Mantık tek yerde kalır.
- **`create_app` neden var, neden modül seviyesinde `app = FastAPI()` yok?**
  Modül import edildiği anda E5 modelini yüklemek, Chroma klasörü açmak istemeyiz.
  Fabrika fonksiyonu sayesinde testler kendi sahte bağımlılıklarını verir, üretimde
  ise `uvicorn study_assistant.api.main:create_app --factory` gerçeklerini kurar.
- **Doğru durum kodları kullanıcıya yol gösterir.** Ollama kapalıysa 500 ve yığın
  izi (stack trace) yerine 503 ve `"Start it with: ollama serve"` mesajı döner;
  arayüz bunu doğrudan bir uyarı bandı olarak gösterebilir.
- **Yarım kayıt olmasın.** Bütün kontroller (uzantı, boyut, sayfa sayısı, taranmış
  PDF) `add_document` çağrısından **önce** yapılır; hata olursa depoya hiçbir şey
  yazılmaz.

## Nasıl çalışıyor?

1. **Yükleme (`upload_document`)**:
   - Dosya adı `.pdf` ile bitmiyorsa (büyük/küçük harf fark etmez) → 400.
   - Baytlar okunur; `max_upload_mb` aşılırsa → 413.
   - `_check_limits` içinde `count_pages`; okunamayan PDF (`InvalidPDFError`) → 400,
     sayfa sayısı `max_pages` üstündeyse → 413.
   - `make_doc_id` içerikten kimlik üretir. `store.has_document` doğruysa aynı
     `DocumentInfo` 200 ile döner (tekrar indeksleme yok).
   - Değilse `_index_pdf`: `extract_pages` (taranmışsa `ScannedPDFError` → 422),
     `chunk_pages(pages, chunk_size, chunk_overlap)`, `store.add_document` → 201.
   - 200/201 ayrımı için endpoint `Response` parametresi alır ve gerekirse
     `response.status_code = 200` yapar; varsayılan `status_code=201` dekoratörde.
2. **Soru (`ask_question`)**: `doc_id` verildiyse ve depoda yoksa 404. Sonra
   `rag.ask(..., k=settings.top_k, min_score=settings.min_score)`; boş soruda
   `ValueError` → 400.
3. **Quiz (`create_quiz`)**: `generate_quiz` çağrılır; `KeyError` → 404,
   `ValueError` (kötü soru sayıları) → 400, `QuizGenerationError` → 502. Cevap
   `quiz.public()` – yani `correct_option` ve `reference_answer` **gizlenir**.
4. **Puanlama (`grade`)**: `QuizRepository.get` ile quiz yüklenir, yoksa
   `"Quiz not found"` 404. `grade_answer` hatalarından `KeyError` → 404,
   `ValueError` (eksik şık seçimi) → 400.
5. **Sağlık (`health`)**: `llm.check()` çalışır; `LLMError` gelirse `llm_ok=False` ve
   mesaj `llm_detail` içine yazılır. Uç nokta **her zaman 200** döner, çünkü API'nin
   kendisi ayakta; sadece LLM durumu raporlanır.
6. **Senkron `def` endpoint'ler**: Embedding, Chroma, SQLite ve Ollama çağrıları
   bloklayıcı. `async def` içinde çağırsaydık olay döngüsü (event loop) kilitlenirdi.
   Düz `def` yazınca FastAPI bunları kendi iş parçacığı havuzunda (threadpool)
   çalıştırır.

## Önemli kavramlar

- **REST**: Kaynaklar (doküman, quiz) URL ile adlandırılır; işlem HTTP yöntemiyle
  ifade edilir (`GET` oku, `POST` oluştur, `DELETE` sil).
- **Durum kodları**: 200 OK, 201 Created (yeni kaynak), 204 No Content (silindi,
  gövde yok), 400 Bad Request (istemci hatası), 404 Not Found, 413 Payload Too Large,
  422 Unprocessable (dosya geçerli ama işlenemiyor: taranmış PDF), 502 Bad Gateway
  (arkadaki servis – LLM – bozuk cevap verdi), 503 Service Unavailable (arkadaki
  servis kapalı).
- **Bağımlılık enjeksiyonu (dependency injection)**: Fonksiyon ihtiyaç duyduğu
  nesneleri kendisi yaratmaz, dışarıdan alır. Burada `create_app` parametreleri ve
  `app.state.deps` bunu sağlıyor; testte `FakeLLM`, üretimde `OllamaClient`.
- **Uygulama fabrikası (application factory)**: Uygulamayı import anında değil, bir
  fonksiyon çağrısıyla kurmak. Her test kendi temiz uygulamasını alır.
- **Pydantic ile doğrulama**: İstek gövdesi `AskRequest` gibi modellere uymazsa
  FastAPI otomatik 422 döner; `response_model` ise cevabın şeklini garanti eder.
- **Exception handler**: `app.add_exception_handler(LLMError, ...)`. Starlette
  hatanın sınıf hiyerarşisinde (MRO) en özel eşleşmeyi seçer; bu yüzden
  `LLMUnavailableError` 503, genel `LLMError` 502 olur.
- **İdempotent yükleme**: Kimlik içerikten (`make_doc_id`) geldiği için aynı dosyayı
  iki kez yüklemek ikinci kopya üretmez.

## Mülakatta sorulabilecek sorular

1. **201 ile 200 arasındaki fark nedir, bu projede ne zaman hangisi dönüyor?**
   201 yeni kaynak oluşturulduğunda; aynı PDF tekrar yüklenince yeni bir şey
   oluşmadığı için 200 ve mevcut `DocumentInfo` dönüyor.
2. **Ollama kapalıyken neden 500 değil de 503 dönüyoruz?** 500 "bizim kodumuz
   bozuk" demek; 503 ise "bağımlı servis şu an yok, sonra tekrar dene" demek.
   Mesajda düzeltme komutu (`ollama serve`) da var.
3. **502 ile 503 farkı?** 503: servis ulaşılamıyor/hazır değil. 502: servise ulaştık
   ama işe yaramaz bir cevap verdi (ör. quiz'de hiç geçerli soru yok).
4. **Neden `async def` yerine `def` kullandınız?** Çağrılar bloklayıcı; `async def`
   içinde olay döngüsünü kilitlerdi. `def` ile FastAPI bunları threadpool'da çalıştırır.
5. **`create_app` fabrikası testleri nasıl kolaylaştırıyor?** Sahte bağımlılıklar
   (`HashEmbedder`, `FakeLLM`, geçici SQLite) parametre olarak verilir; model
   indirme, ağ veya Ollama gerekmez.
6. **Quiz cevap anahtarının öğrenciye sızmasını nasıl engelliyorsunuz?** `/quiz`
   her zaman `quiz.public()` döner; `response_model=QuizPublic` bu alanları zaten
   içermez. Doğru cevap sadece `/grade` sonrasında gösterilir.
7. **Büyük veya taranmış bir PDF yüklenirse depoda yarım veri kalır mı?** Hayır;
   boyut, sayfa sayısı ve metin katmanı kontrolleri `add_document` öncesinde yapılır.
8. **Health endpoint'i neden LLM kapalıyken bile 200 dönüyor?** API sürecinin
   kendisi sağlıklı; arayüz `llm_ok` alanına bakıp kullanıcıya uyarı gösterir.
   Yük dengeleyici de API'yi gereksiz yere "ölü" saymaz.
