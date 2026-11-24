# 13 – Docker ve yerel çalıştırma betikleri

Bu not, uygulamayı iki şekilde çalıştırmayı anlatıyor: Docker Compose ile konteynerlerde
ve `scripts/run_local.sh` ile doğrudan Mac'te. Hangi parçanın neden konteynerde, hangisinin
neden host'ta kaldığını da açıklıyor.

---

## Ne yaptık?

- **`Dockerfile`**: çok aşamalı (multi-stage) bir imaj. Birinci aşama
  (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`) bağımlılıkları `uv sync --locked`
  ile kurar. İkinci aşama (`python:3.12-slim-bookworm`) yalnızca hazır `/app` dizinini
  alır ve `app` adlı, uid 1000 bir kullanıcıyla çalışır.
- **`docker-compose.yml`**: aynı imajdan iki servis. `api` FastAPI'yi (`:8000`), `ui`
  Streamlit'i (`:8501`) başlatır. `api` için `/health` kontrolü var; `ui` ancak API
  sağlıklı olduğunda açılır.
- **`.dockerignore`**: `.git`, `.venv`, `data/`, `tests/`, `evals/`, önbellekler ve
  `.env` gibi gizli dosyalar imaj bağlamına girmez.
- **`scripts/run_local.sh`**: Ollama'nın çalıştığını ve modelin çekildiğini kontrol eder,
  API ile UI'yi arka planda başlatır, sağlık uç noktasını bekler ve Ctrl+C ile ikisini
  birden durdurur. `make start` bu betiği çağırır.

## Neden?

- **Ollama neden konteynerde değil?** macOS'taki Docker Desktop, konteynerlere GPU
  erişimi vermez. Ollama host'ta GPU (Apple Silicon'da Metal) ile çalışır; konteynerde
  olsaydı yalnızca CPU ile yavaş çalışırdı. Konteynerler `host.docker.internal:11434`
  üzerinden host'taki Ollama'ya bağlanır.
- **Neden iki aşama?** Derleme aracı (uv) ve bağımlılıkların derleme çıktısı son imaja
  girmez. Son imaj daha küçük ve daha az saldırı yüzeyine sahip olur.
- **Neden `pyproject.toml` ve `uv.lock` önce kopyalanıyor?** Docker katmanları
  önbelleğe alır. Kaynak kodu değiştiğinde yalnızca son `uv sync` yeniden çalışır;
  bağımlılıklar yeniden indirilmez.
- **Neden root değil?** Konteyner içinde bir açık olursa saldırgan root yetkisi
  kazanmasın diye. `app` kullanıcısı yalnızca `/app` içine yazabilir.
- **Neden `hf-cache` adlı bir volume?** Gömme modeli (`multilingual-e5-small`) ilk
  açılışta indirilir ve yüzlerce MB tutar. Volume sayesinde her `up` işleminde yeniden
  indirilmez.

## Nasıl çalışıyor?

1. `docker compose up --build` Dockerfile'ı iki aşamada derler. Builder aşamasında
   `/app/.venv` oluşur; runtime aşamasında bu dizin kopyalanır.
2. Compose `api` konteynerini başlatır. Sağlık kontrolü `127.0.0.1:8000/health`
   adresine Python'un `urllib` modülüyle istek atar; `start_period: 120s` gömme modelinin
   yüklenmesi için zaman tanır.
3. Sağlık kontrolü geçince `ui` başlar. `STUDY_API_URL=http://api:8000` ile API'ye
   compose ağındaki servis adıyla bağlanır.
4. `scripts/run_local.sh` aynı sırayı host'ta yapar: önce `curl` ile Ollama'yı, sonra
   `ollama list` ile modeli kontrol eder. `uv sync` ardından `.venv/bin/uvicorn` ve
   `.venv/bin/streamlit` doğrudan çalıştırılır; böylece durdurulan PID gerçek sunucunun
   PID'idir. Sağlık beklemesi `/health` 200 dönene kadar sürer (en fazla 120 sn).
5. Ctrl+C (INT) veya TERM sinyali `trap` ile yakalanır, `exit` çağrılır ve `EXIT`
   tuzağı çocuk süreçleri öldürür.

## Önemli kavramlar

- **İmaj ve konteyner:** İmaj salt okunur bir şablondur (katmanlardan oluşur).
  Konteyner, o imajdan başlatılmış çalışan bir örnektir. Aynı imajdan birden fazla
  konteyner çalışabilir; burada `api` ve `ui` böyle.
- **Çok aşamalı derleme (multi-stage build):** Birden fazla `FROM` satırı; yalnızca son
  aşama imaja girer. `COPY --from=builder` ile ara çıktı taşınır.
- **Katman önbelleği:** Bir satır değişince o satırdan sonraki tüm katmanlar yeniden
  çalışır. Sık değişen kaynak kodunu en sona koymak bu yüzden önemli.
- **`host-gateway`:** Linux'ta `host.docker.internal` adı otomatik tanımlı değildir;
  `extra_hosts` ile ana makinenin IP'sine eşlenir. macOS'ta Docker Desktop zaten sağlar.
- **Bind mount ve named volume:** `./data:/app/data` host'taki bir klasörü bağlar;
  `hf-cache:` Docker'ın yönettiği bir depolama alanıdır. İlk oluşturulurken imajdaki
  dizinin sahipliğini devralır.
- **`depends_on` + `service_healthy`:** Yalnızca "başladı" değil, "sağlıklı" koşulunu
  bekler.
- **Sinyaller ve `trap`:** Ctrl+C bir `SIGINT` gönderir. Betik bunu yakalamazsa çocuk
  süreçler açık kalır. Arka plan işleri (`&`) etkileşimsiz bir kabukta SIGINT'i
  yoksayar; bu yüzden testte SIGTERM kullandık.

## Mülakatta sorulabilecek sorular

- Konteyner ile sanal makine arasındaki fark nedir?
- Multi-stage build neden imaj boyutunu küçültür? Bir örnek verin.
- Katman önbelleğini verimli kullanmak için Dockerfile'da satır sırası nasıl olmalı?
- Konteyner neden root olarak çalıştırılmamalı? `USER` satırı ne yapar?
- macOS'ta Ollama'yı neden Docker içinde çalıştırmıyoruz? Bunun bedeli nedir?
- `depends_on` tek başına ne garanti eder, `service_healthy` ne ekler?
- Bir shell betiğinde Ctrl+C geldiğinde arka plan süreçlerini nasıl temiz kapatırsınız?
- Named volume ile bind mount arasındaki fark nedir; hangisini ne zaman seçersiniz?
