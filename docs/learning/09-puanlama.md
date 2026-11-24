# 09 – Cevap puanlama: deterministik çoktan seçmeli, LLM'li açık uçlu

Bu not, öğrencinin quiz cevabını nasıl puanladığımızı anlatıyor. İki tür soru var ve
ikisi farklı yollardan geçiyor: çoktan seçmeli sorular **kod** ile, açık uçlu sorular ise
**yerel LLM** ile (LLM-as-judge) puanlanıyor.

---

## Ne yaptık?

- **`src/study_assistant/grading.py`** yazdık:
  - `verdict_for(score)`: 0–100 puanı etikete çevirir. `>= 80` → `correct`,
    `>= 40` → `partially_correct`, aksi halde `incorrect`.
  - `grade_answer(quiz, question_id, store, llm, *, answer=None, selected_option=None)`:
    asıl giriş noktası. Önce soruyu bulur (yoksa `KeyError`), sonra türüne göre dallanır.
  - Çoktan seçmeli dal: `selected_option` yoksa veya `0..len(options)-1` dışındaysa
    `ValueError`. Doğru şıkla eşleşirse 100, değilse 0 puan. LLM **çağrılmaz**.
  - Açık uçlu dal: boş cevap (`None`, `""` ya da sadece boşluk) için 0 puan ve
    `"No answer given."` geri bildirimi, yine LLM yok. Dolu cevapta kaynak parçaları
    (`store.get_chunks(doc_id, source_pages)`) ve soru, referans cevap, öğrenci cevabı
    `build_grade_prompt` ile tek bir kullanıcı mesajına birleştirilir ve `GRADE_SYSTEM`
    ile `LLMGrade` şemasında LLM'e gönderilir.
  - Açık uçlu puan `max(0, min(100, score))` ile sınırlanır; etiketi **her zaman**
    `verdict_for` hesaplar, modelin söylediği etiket dikkate alınmaz.
- **`tests/test_grading.py`**: çoktan seçmeli doğru/yanlış, seçim eksikliği ve aralık
  dışı seçim, puan sınırlama, boş cevap, eşik değerleri, bilinmeyen soru ve prompt
  kurulumu için testler. LLM yerine `FakeLLM`, embedding yerine `HashEmbedder`
  kullanılıyor; ağa ve Ollama'ya hiç gidilmiyor.

## Neden?

**Neden çoktan seçmelide LLM yok?** Çoktan seçmeli soruda doğru cevap zaten bellidir:
`correct_option` alanı. Bunu bir modele sormanın iki kötü tarafı var. Birincisi, model
bazen yanlış şıkkı doğru sanabilir; ikincisi, aynı cevap iki kez sorulduğunda farklı puan
verebilir. Kodla karşılaştırma **deterministiktir**: aynı girdi her zaman aynı puanı
verir, ücretsizdir ve yerel modelin yavaşlığını beklemek gerekmez.

**Neden açık uçlu cevapta LLM?** "Glikozun enerjiye dönüştürülmesi" ile "Glikoz
parçalanıp ATP üretilir" aynı anlamı taşıyabilir ama kelime kelime eşleşmez. Bu yüzden
anlamsal bir karşılaştırma gerekir; bunu yapabilecek tek bileşen LLM'dir.

**Neden puanı ve etiketi biz hesaplıyoruz?** Model `score` ve `feedback` alanlarını
döndürür, ama `verdict` alanını döndürmez. Eşik kuralını (80/40) tek bir yerde tutmak,
değiştirmek ve test etmek kolay olsun diye etiketi `verdict_for` hesaplar. Model
150 gibi aralık dışı bir sayı da üretebilir; bu yüzden puan `0..100` aralığına
sıkıştırılır (`clamp`).

**Neden boş cevabı LLM'e sormuyoruz?** Boş bir cevabı değerlendirmenin bilgi değeri yok,
ama LLM'e gönderirsek model yine de bir şey uydurabilir ve 20 gibi bir puan verebilir.
Boş cevap için sonuç kesin olduğundan kodla 0 veriyoruz.

**Neden `chunks[:4]`?** Yerel model 8k token penceresine sahip ve prompt kısa tutulmalı.
Puanlama için dört kaynak parçası çoğu zaman yeterli.

## Nasıl çalışıyor?

Çoktan seçmeli akış:

```
grade_answer(quiz, "q1", store, llm, selected_option=1)
  -> _find_question             # KeyError yoksa soru bulunur
  -> _citations                 # source_pages -> Citation listesi
  -> _grade_multiple_choice
       selected_option == correct_option ? 100 / 0
       feedback: "Correct! The answer is: ..." / "Not quite. The correct answer is: ..."
```

Açık uçlu akış:

```
grade_answer(quiz, "q2", store, llm, answer="Glikoz enerjiye dönüşür.")
  -> text = answer.strip()
  -> boşsa: Grade(score=0, verdict="incorrect", feedback="No answer given.")
  -> chunks = store.get_chunks(doc_id, source_pages)
  -> prompt = build_grade_prompt(question, chunks, text)
       "[1] (dosya.pdf, s. 2)\n...\n\nQuestion: ...\nReference answer: ...\nStudent answer: ..."
  -> llm.complete_json(GRADE_SYSTEM, prompt, LLMGrade)   # {"score": int, "feedback": str}
  -> score = clamp(0..100); verdict = verdict_for(score)
```

`LLMClient.complete_json` yanıtı Pydantic ile doğrular ve bozuk JSON'da bir kez yeniden
dener; ikinci denemede de bozuksa `LLMError` fırlatır. Bu hata yukarıya, API katmanına
gider. Yani `grade_answer` içinde JSON ayrıştırma kodu yok; bu sorumluluk `llm.py`'de.

Sayfa atıfları (`Citation`) her iki tür için de aynıdır: soru hangi sayfalardan üretildiyse
o sayfaların listesi döner. Böylece arayüz "bu cevap 2. ve 3. sayfadan" diyebilir.

## Önemli kavramlar

- **LLM-as-judge (LLM hakem olarak):** Bir dil modelinin başka bir cevabın kalitesini
  puanlaması. Esnek ama tutarsız olabilir; bu yüzden girdiyi sınırlar, referans cevap
  verir ve çıktıyı şemayla zorlarız.
- **Referans cevap (reference answer):** Quiz üretilirken kaynak metinden çıkarılmış
  "altın" cevap. Modelin puanlaması bu cevaba ve kaynaklara dayanır, serbest bilgiye
  değil.
- **Deterministik puanlama:** Aynı girdi için her zaman aynı çıktı. Çoktan seçmelide
  bunu sağlamak mümkün, açık uçluda mümkün değil.
- **Clamp (sınırlama):** Bir değeri belirli bir aralığa zorlamak. `max(0, min(100, x))`.
- **Eşik (threshold) ile etiketleme:** Sayısal puanı kategoriye çevirme. Eşikler
  bir fonksiyonda toplanır; böylece sınır değerleri test edilebilir (80, 79, 40, 39).
- **Savunmacı doğrulama:** Model çıktısına güvenmeme. Etiketi model değil biz
  hesaplarız, puanı sınırlarız, boş cevabı modele sormayız.
- **Atıf (citation):** Cevabın dayandığı kaynak sayfa. Öğrencinin kendisi kontrol
  edebilmesi için gereklidir.

## Mülakatta sorulabilecek sorular

1. **LLM'e neden her şeyi değerlendirtmiyoruz?** Maliyet, gecikme ve tutarsızlık.
   Kesin cevabı olan sorularda kod daha güvenilirdir; sadece anlamsal karşılaştırma
   gereken yerde LLM kullanılır.
2. **Modelin `verdict` döndürmesine izin verseydik ne olurdu?** Eşik kuralı modelin
   içine gömülür, değiştirmek için prompt'u yeniden yazmak gerekirdi; ayrıca model
   puanla etiketi birbirine uymayan çıktılar üretebilirdi.
3. **LLM 150 puan döndürürse ne yapılır?** Pydantic şemasında kısıt koymadık (küçük
   modeller için şema basit tutuldu); kısıt uygulamada `max(0, min(100, ...))` ile
   kodda sağlanır.
4. **Neden boş cevabı LLM'e göndermiyoruz?** Gereksiz çağrı, model halüsinasyonu riski
   ve sonucun zaten kesin olması.
5. **Puanlamayı nasıl test edersiniz, LLM olmadan?** `FakeLLM` ile önceden hazırlanmış
   yanıtlar vererek; ayrıca `llm.calls` listesi ile LLM'in çağrılıp çağrılmadığını
   doğrulayarak.
6. **Çoktan seçmelide `selected_option` yoksa neden `ValueError`?** Bu bir istemci
   hatasıdır; API katmanı bunu 422 gibi bir yanıta çevirebilir. Sessizce 0 puan vermek
   öğrenciyi yanıltır.
7. **Açık uçlu sorularda kaynak olarak neden tüm chunk'ları değil de yalnızca
   `source_pages` sayfalarını alıyoruz?** Hem bağlam penceresini küçük tutmak hem de
   puanlamayı quiz sorusunun gerçekten dayandığı metinle sınırlamak için.
