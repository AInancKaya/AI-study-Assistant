# 01 – Proje İskeleti (uv, src layout, Pydantic, ayarlar, CI)

Bu not, projenin ilk adımında kurduğumuz "iskeleti" anlatıyor. Henüz PDF okumuyoruz,
soru cevaplamıyoruz; ama bundan sonraki her parça bu temelin üstüne oturacak.

---

## Ne yaptık?

- **`uv` ile Python projesi oluşturduk.** Python 3.12'yi `.python-version` dosyasıyla
  sabitledik, bağımlılıkları `pyproject.toml`'a yazdık ve tam sürümlerini `uv.lock`
  dosyasına kilitledik.
- **`src/` düzenini (src layout) kullandık.** Kodumuz `src/study_assistant/` altında,
  testler `tests/` altında.
- **`config.py`** yazdık: model adı, Ollama adresi, chunk boyutu gibi tüm ayarlar tek
  bir `Settings` sınıfında. Ortam değişkenleriyle (`STUDY_...`) değiştirilebiliyor.
- **`models.py`** yazdık: uygulamanın her katmanının konuşacağı veri tipleri
  (`Page`, `Chunk`, `Answer`, `Quiz`, `Grade`, ve LLM'in döndüreceği `LLM*` şemaları).
- **Araçları kurduk:** `ruff` (lint + format), `pytest` (test + coverage),
  `pre-commit` (commit öncesi kontrol), GitHub Actions (her push'ta otomatik test).
- **`Makefile`** ekledik: `make setup`, `make test`, `make lint`, `make run` gibi kısa
  komutlar.
- MIT lisansı, `.env.example` ve geçici bir README ekledik.

## Neden?

- **Tekrarlanabilirlik:** "Benim bilgisayarımda çalışıyordu" sorununu önlemek için.
  `uv.lock` sayesinde sen, ben ve CI sunucusu birebir aynı paket sürümlerini kurarız.
- **Erken hata yakalama:** Lint, format ve testler daha ilk günden çalışırsa, hatalar
  büyümeden yakalanır. Sonradan eklemek her zaman daha zahmetlidir.
- **Tek doğruluk kaynağı:** Model adı ya da `min_score` gibi sayılar kodun içine
  dağılırsa, birini değiştirmek için on dosyayı aramak gerekir. Hepsi `config.py`'de.
- **Sözleşme önce gelir:** `models.py` diğer modüllerin birbirine ne göndereceğini
  baştan tanımlar. Böylece PDF okuyucu, arama motoru ve API ayrı ayrı yazılsa bile
  birbirine uyar.
- **Sıfır maliyet:** Ücretli API yok, API anahtarı yok. LLM yerelde Ollama ile çalışacak.

## Nasıl çalışıyor?

### uv

`uv`, Rust ile yazılmış çok hızlı bir Python paket ve proje yöneticisi. `pip`, `venv`,
`pyenv` ve `pip-tools`'un işini tek başına yapar:

```bash
uv python install 3.12        # Python'u kurar
uv init --lib .               # pyproject.toml + src/ iskeleti
uv add fastapi                # bağımlılık ekler, uv.lock'u günceller
uv add --dev pytest           # sadece geliştirmede gereken paket
uv run pytest                 # .venv'i otomatik kurup komutu onun içinde çalıştırır
uv sync --locked              # CI'da: lock dosyasına birebir uy, değiştirme
```

`uv run` sayesinde sanal ortamı elle "activate" etmek gerekmez.

### src layout

```
src/study_assistant/   ← paket kodu
tests/                 ← testler
```

Kod doğrudan proje kökünde olsaydı, testler yanlışlıkla "kurulmamış" klasörü import
edebilirdi. `src/` düzeninde paket önce kurulur, testler de kurulu paketi kullanır;
yani testler kullanıcının göreceği hâli test eder.

### pyproject.toml

Projenin kimlik kartı: ad, sürüm, Python sürümü, bağımlılıklar ve araç ayarları
(`[tool.ruff]`, `[tool.pytest.ini_options]`) tek dosyada.

### Pydantic modelleri

```python
class Citation(BaseModel):
    doc_id: str
    filename: str
    page: int
```

Pydantic, tip ipuçlarını (type hints) kullanarak veriyi **doğrular**. `page="abc"`
verirsen hata alırsın. Ayrıca `model_dump()` ile sözlüğe, `model_json_schema()` ile
JSON şemasına çevirebilir. Bu şemayı ileride Ollama'ya verip "bana tam bu formatta
JSON döndür" diyeceğiz.

İki önemli tasarım kararı:

1. **`Quiz.public()`**: Öğrenciye gönderilen quiz'de doğru cevap
   (`correct_option`) ve örnek cevap (`reference_answer`) **olmamalı**. `public()`
   bunları çıkarıp `QuizPublic` döndürür. Testimiz tam olarak bunu kontrol ediyor.
2. **`LLM*` şemalarında sayısal kısıt yok** (`minimum`, `maxLength` vb.). Küçük yerel
   modeller karmaşık şemalarda daha sık hata yapar. Aralık kontrollerini (ör. puan
   0–100 mü?) LLM'e değil, kendi Python kodumuza yaptıracağız.

### pydantic-settings ve ortam değişkenleri

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDY_", env_file=".env")
    llm_model: str = "gemma2:9b"
```

Varsayılan değer `gemma2:9b`. Ama terminalde

```bash
STUDY_LLM_MODEL=qwen3:8b make api
```

yazarsan kod değişmeden başka model kullanılır. `.env` dosyası da aynı işi görür
(`.gitignore`'da olduğu için gizli kalır; örneği `.env.example`'da).

`get_settings()` fonksiyonu `@lru_cache` ile sarılı: ayarlar bir kez okunur, sonra hep
aynı nesne döner. Testlerde ise `conftest.py` içindeki bir fixture her testten önce bu
önbelleği temizliyor; yoksa bir testin değiştirdiği ayar diğerine sızabilirdi.

### ruff

Hem linter (hatalı/şüpheli kod bulur) hem formatter (kodu tek tip biçimlendirir).
Seçtiğimiz kurallar: `E`/`F` (temel hatalar), `I` (import sırası), `B` (sık yapılan
bug'lar), `UP` (modern Python sözdizimi), `SIM` (sadeleştirme önerileri).

```bash
uv run ruff check .            # sorunları listele
uv run ruff format .           # kodu biçimlendir
```

### pytest ve marker'lar

İki özel işaret (marker) tanımladık:

- `@pytest.mark.slow` → gerçek embedding modelini indiren/çalıştıran testler
- `@pytest.mark.live` → çalışan bir Ollama sunucusu isteyen testler

`pyproject.toml`'daki `addopts = "-m 'not slow and not live' ..."` sayesinde düz
`uv run pytest` bunları **atlar**; hızlı ve internetsiz çalışır. İstediğimizde
`make eval` / `make live-eval` ile ayrıca çalıştırırız. `--cov` ile de kodun yüzde
kaçının testlerle çalıştırıldığını (coverage) görürüz.

### pre-commit

`git commit` yazdığında, commit oluşmadan önce `ruff`, `ruff-format` ve `uv-lock`
kancalarını (hook) çalıştırır. Biçimsiz kod veya güncellenmemiş lock dosyası depoya
hiç girmez. Kurulum: `uv run pre-commit install`.

### GitHub Actions (CI)

`.github/workflows/ci.yml` her push ve pull request'te bir Ubuntu makinesinde şunu
yapar: `uv sync --locked` → `ruff check` → `ruff format --check` → `pytest`. Biri
başarısız olursa GitHub'da kırmızı çarpı görürsün. Hiçbir gizli anahtar (secret)
gerekmiyor, çünkü testler dış servise bağlanmıyor.

### Testlerde neden "fake" kullanacağız?

İleriki görevlerde Ollama ve embedding modeli için **sahte (fake)** sınıflar
yazacağız. Sebebi:

- **Hız:** Gerçek LLM bir cevap için saniyeler sürer; fake milisaniyeler.
- **Belirlilik (determinism):** LLM her seferinde farklı cevap verebilir; fake hep aynı
  cevabı verir, test ya hep geçer ya hep kalır.
- **CI'da çalışabilmek:** GitHub'ın makinesinde Ollama yok, 9B'lik model de yok.
- **Hata senaryolarını test etmek:** "LLM bozuk JSON döndürürse ne olur?" sorusunu
  gerçek modelle tetiklemek zor; fake'e bozuk JSON döndürtmek tek satır.

Gerçek modelle uçtan uca kontrolü ise `live` işaretli testler yapacak.

## Önemli kavramlar

| Kavram | Kısa açıklama |
| --- | --- |
| Lock dosyası (`uv.lock`) | Tüm paketlerin (alt bağımlılıklar dahil) tam sürümleri |
| Sanal ortam (`.venv`) | Projeye özel, sistemden izole Python paketleri |
| src layout | Paket kodunu `src/` altına koyup testlerin kurulu paketi kullanması |
| Pydantic `BaseModel` | Tip ipuçlarıyla doğrulama + JSON dönüşümü yapan veri sınıfı |
| JSON Schema | Bir JSON'un yapısını tarif eden standart; LLM'e format dayatmak için |
| `BaseSettings` | Ayarları ortam değişkenlerinden / `.env`'den okuyan Pydantic sınıfı |
| `lru_cache` | Fonksiyon sonucunu önbelleğe alır; ayarlar bir kez okunur |
| Fixture | pytest'te testlere hazırlık/temizlik sağlayan fonksiyon |
| Marker | Testleri etiketleyip seçerek çalıştırma (`-m slow`) |
| Coverage | Testlerin kodun ne kadarını çalıştırdığının ölçüsü |
| Linter / Formatter | Kod hatası arayan / kodu biçimlendiren araç (ikisi de ruff) |
| CI | Her push'ta testleri otomatik çalıştıran sistem |
| Fake | Gerçek bağımlılığın yerine geçen basit, kontrollü sahte sınıf |
| TDD | Önce başarısız test yaz, sonra geçecek kadar kod yaz |

## Mülakatta sorulabilecek sorular

1. **`pip install` ile `uv add` arasındaki fark nedir? Lock dosyası neden önemli?**
   `uv add` paketi `pyproject.toml`'a yazar ve `uv.lock`'u günceller; böylece herkes
   aynı sürümleri kurar. `pip install` sadece o anki ortama kurar, kayıt tutmaz.
2. **src layout'un avantajı nedir?** Testler kurulu paketi import eder; paketlemeyi
   unuttuğun bir dosya varsa testte fark edersin.
3. **Ayarları neden ortam değişkenlerinden okuyoruz?** 12-Factor App ilkesi: kod aynı
   kalır, ortam (geliştirme/test/üretim) değişir. Gizli bilgiler koda girmez.
4. **Pydantic ne işe yarar, `dataclass`'tan farkı ne?** Pydantic çalışma zamanında
   doğrulama ve dönüşüm yapar, JSON şeması üretir; `dataclass` doğrulama yapmaz.
5. **Öğrenciye giden quiz'den cevapları nasıl ve neden ayırdınız?** Ayrı bir
   `QuizPublic` modeli ve `Quiz.public()` metodu ile. Sadece alanı gizlemek yerine ayrı
   tip kullanmak, cevabın yanlışlıkla API'den sızmasını tip düzeyinde engeller.
6. **LLM çıktı şemasında neden `minimum`/`maxLength` gibi kısıtlar yok?** Küçük yerel
   modeller bu kısıtlara güvenilir şekilde uymaz; kontrolü kodda yapıp hatalı öğeleri
   eleyerek sistemi daha sağlam yaparız.
7. **Birim testlerde gerçek LLM yerine neden fake kullanılır?** Hız, belirlilik,
   CI'da çalışabilme ve hata senaryolarını kolayca üretebilme.
8. **`lru_cache` ile önbelleğe alınmış ayarlar testlerde ne sorun çıkarabilir?**
   Bir testteki değişiklik diğerine sızar; bu yüzden her testte `cache_clear()` yaptık.
9. **CI'da neden `uv sync --locked` kullanılır?** Lock dosyası `pyproject.toml` ile
   uyumsuzsa hata verir; "lock'u güncellemeyi unuttum" hatasını yakalar.
