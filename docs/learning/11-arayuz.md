# 11 – Streamlit arayüzü: API'yi tarayıcıya bağlamak

Bu not, projenin son parçasını anlatıyor: `ui/app.py` ile bir Streamlit sayfası
yazıp onu `ui/api_client.py` üzerinden FastAPI backend'ine bağladık. Arayüz hiçbir
modeli, vektör deposunu ya da Ollama'yı doğrudan çağırmaz; yalnızca HTTP konuşur.

---

## Ne yaptık?

- **`ui/api_client.py`**: `ApiClient` sınıfı, API'nin her ucu için bir metot içerir
  (`health`, `upload`, `documents`, `delete`, `ask`, `quiz`, `grade`). Metotlar
  ayrıştırılmış JSON döndürür.
- **`ApiError`**: her hata tek bir istisna tipine çevrilir. `status` ve `detail`
  alanları vardır. API'ye hiç ulaşılamazsa `status=0` ve "Run: make api" ipucu.
- **`ui/app.py`**: tek sayfa, iki sekme.
  - Kenar çubuğu: LLM durumu, PDF yükleme, not listesi ve silme.
  - **Ask**: sohbet arayüzü (`st.chat_input` / `st.chat_message`). Cevaplar
    `📄 dosya.pdf · p. 4` biçiminde sayfa rozetleriyle gösterilir.
  - **Quiz**: quiz üret, çoktan seçmeli sorular için radyo düğmesi, açık uçlu
    sorular için metin alanı, her soru için "Check" düğmesi. Sonuçta puan,
    doğruluk etiketi, geri bildirim, referans cevap ve sayfa rozetleri çıkar.
- **`tests/test_api_client.py`**: `httpx.MockTransport` ile 7 test. Ağ yok.

## Neden?

- **İstemci ile arayüzü ayırmak.** HTTP ayrıntıları (yol, JSON gövdesi, hata
  biçimi) tek dosyada kalır. Sayfa kodu `client.ask(...)` gibi düz çağrılar yapar
  ve testleri ağ olmadan yazılabilir.
- **Tek istisna tipi.** Arayüz her hatada `except ApiError` yazar ve
  `st.error(exc.detail)` ile gösterir. Yığın izi (traceback) kullanıcıya gitmez.
- **`transport` parametresi.** Testler gerçek bir sunucu yerine `MockTransport`
  verir. Üretimde parametre boş kalır ve gerçek ağ kullanılır.
- **`st.cache_resource`.** `ApiClient` (ve içindeki `httpx.Client` bağlantı
  havuzu) her yeniden çalıştırmada yeniden kurulmaz; süreç boyunca bir tane yaşar.
- **Sohbet geçmişi `st.session_state`'te.** Streamlit her etkileşimde betiği baştan
  çalıştırır. Değişkenler kaybolur, `session_state` ise oturum boyunca kalır.

## Nasıl çalışıyor?

1. **Betik her etkileşimde baştan çalışır.** Düğmeye basınca `main()` yeniden
   çağrılır. Kalıcı olması gereken her şey (sohbet, aktif quiz, verilen puanlar)
   `st.session_state` içinde tutulur.
2. **Soru akışı**: `render_ask` önce `chat_input`'u kontrol eder. Yeni soru varsa
   `_answer_question` hem soruyu hem API cevabını geçmişe ekler, sonra geçmiş
   baştan çizilir. Bu sıra sayesinde yeni mesaj aynı çalıştırmada görünür.
3. **Quiz akışı**: "Generate quiz" üretilen quizi `session_state["quiz"]`'e koyar
   ve puanları sıfırlar. Her "Check" tıklamasında `_check` cevabı `/grade`'e
   gönderir ve sonucu `session_state["grades"][question_id]` içine yazar.
4. **Hata akışı**: her API çağrısı `_call` ya da `_delete_document` üzerinden geçer.
   `ApiError` yakalanır, sayfada bir `st.error` olarak görünür, uygulama çökmez.
5. **Import yolu**: `streamlit run ui/app.py` komutu `ui/` klasörünü `sys.path`'e
   ekler, repo kökünü değil. Bu yüzden `app.py` başında kökü kendisi ekler; testler
   için ise `pyproject.toml` içinde `pythonpath = ["."]` vardır.

## Önemli kavramlar

- **HTTP istemcisi ve bağlantı yeniden kullanımı**: `httpx.Client` bir kez
  oluşturulur ve her istekte aynı bağlantı havuzu kullanılır.
- **`MockTransport`**: `httpx`'in taşıma katmanını taklit eden bir nesne. Gelen
  isteği kaydedip önceden hazırlanmış bir cevap döndürür.
- **Multipart yükleme**: dosya, `Content-Type: multipart/form-data` gövdesinde
  `file` alanı olarak gönderilir.
- **Durum bilgisi (state) ve yeniden çalıştırma**: Streamlit'te her etkileşim betiği
  baştan çalıştırır; durum `st.session_state` ile taşınır.
- **Widget anahtarları (`key`)**: quiz id ve soru id'si anahtara eklenir. Böylece yeni
  bir quiz eski radyo ve metin alanlarının değerini devralmaz.
- **`st.cache_resource` ve `st.cache_data`**: `cache_resource` nesneler (istemci,
  bağlantı) için, `cache_data` serileştirilebilir veriler için kullanılır.
- **Hata sözleşmesi**: API'nin `{"detail": ...}` gövdesi istemcide tek bir metne
  indirgenir. FastAPI 422 hatalarında `detail` bir liste olur; `_error_detail` onu
  okunabilir metne çevirir.

## Mülakatta sorulabilecek sorular

1. Streamlit betiği neden her etkileşimde baştan çalışır ve bu durum durumu
   (state) nasıl yönetmemizi gerektirir?
2. `st.cache_resource` ile `st.cache_data` arasındaki fark nedir? `ApiClient` için
   hangisini seçtik ve neden?
3. Arayüz neden doğrudan Ollama ya da Chroma'yı çağırmıyor? Bu ayrımın faydası ve
   bedeli nedir?
4. `ApiError` içinde `status=0` ne anlama geliyor? Ağ hatasını HTTP hatasından nasıl
   ayırt edersin?
5. `httpx.MockTransport` ile gerçek bir sunucuyu ayağa kaldırmak arasında hangi
   durumda hangisi tercih edilir?
6. Quiz cevap anahtarı neden API'nin ilk cevabında gönderilmiyor ve `/grade`
   çağrısı bu tasarımı nasıl destekliyor?
7. Bir kullanıcı iki kez "Check" basarsa ne olur? Bu durumu nasıl ele alırdın?
