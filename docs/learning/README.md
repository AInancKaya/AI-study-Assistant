# Öğrenme notları

Bu proje adım adım geliştirildi ve her adım için bir not yazıldı. Notlar, bir RAG tabanlı ders asistanının nasıl yapıldığını ve neden böyle yapıldığını anlatır. Sırayla okumanız önerilir.

1. [Proje iskeleti](01-proje-iskeleti.md): uv ile Python 3.12 projesi, Pydantic ayarları, CI.
2. [PDF okuma](02-pdf-okuma.md): sayfa metnini çıkarma, Türkçe karakterler ve satır sonu tireleri.
3. [Parçalama](03-parcalama.md): sayfa sınırını aşmayan, örtüşmeli parçalar (chunk).
4. [Embedding](04-embedding.md): metni vektöre çevirme ve e5 modeli.
5. [Vektör veritabanı](05-vektor-veritabani.md): Chroma ile parçaları saklama ve benzerlik araması.
6. [Yerel LLM](06-yerel-llm-ollama.md): Ollama ile JSON çıktısı ve hata yönetimi.
7. [RAG](07-rag.md): kaynaklara dayalı cevap ve doğrulanmış sayfa atıfları.
8. [Quiz üretimi](08-quiz-uretimi.md): soru üretimi, doğrulama ve SQLite'ta saklama.
9. [Puanlama](09-puanlama.md): çoktan seçmelisi deterministik, açık uçlusu LLM ile.
10. [FastAPI](10-fastapi.md): çekirdek modülleri HTTP API'sine açmak.
11. [Arayüz](11-arayuz.md): Streamlit arayüzünü API'ye bağlamak.
12. [Değerlendirme](12-degerlendirme.md): sistemin gerçekten işe yarayıp yaramadığını ölçmek.
13. [Docker](13-docker.md): Docker imajı ve yerel çalıştırma betikleri.

Her not şu bölümlerden oluşur: Ne yaptık? · Neden? · Nasıl çalışıyor? · Önemli kavramlar · Mülakatta sorulabilecek sorular.

Mimari genel bakış için bkz. [../architecture.md](../architecture.md).
