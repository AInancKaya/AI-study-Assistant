# 07 – RAG: Notlara dayalı soru cevaplama ve doğrulanmış kaynaklar

Bu not, önceki görevlerde kurduğumuz parçaları (PDF okuma, parçalama, embedding, vektör
veritabanı, yerel LLM) birleştirip öğrencinin sorusunu **kendi ders notlarından**, sayfa
numarasıyla cevaplayan hattı anlatıyor.

---

## Ne yaptık?

- **`src/study_assistant/prompts.py`** yazdık:
  - `ANSWER_SYSTEM`: soru cevaplama için sistem mesajı (kurallar aşağıda).
  - `QUIZ_SYSTEM` ve `GRADE_SYSTEM`: sonraki görevlerde (quiz üretme ve notlama)
    kullanılacak sistem mesajları. Hepsini tek dosyada topladık ki promptlar kodun
    içine dağılmasın.
  - `format_sources(chunks)`: parçaları numaralı kaynak bloklarına çevirir:

    ```text
    [1] (turkce_notlar.pdf, s. 2)
    Işık ılık süt içinde ağır ağır yayıldı; ...

    [2] (turkce_notlar.pdf, s. 1)
    Fotosentez, bitkilerin ...
    ```

    `s.` = sayfa. Numaralar 1'den başlar ve verilen sırayı korur.
- **`src/study_assistant/rag.py`** yazdık:
  - `NOT_FOUND_MESSAGE`: "I couldn't find this in your notes. / Bu bilgiyi notlarında
    bulamadım." (iki dilli sabit mesaj).
  - `ask(question, store, llm, *, k, min_score, doc_id=None) -> Answer`: asıl fonksiyon.
  - `build_user_prompt(question, sources)`: kaynak bloğu + `Question: <soru>`.
  - `valid_citations(cited, sources)`: LLM'in verdiği kaynak numaralarını doğrulayıp
    `Citation(doc_id, filename, page)` listesine çevirir.
- **`tests/test_rag.py`**: 14 test. Gerçek bir geçici `VectorStore` (`HashEmbedder` ile)
  ve `tests/fixtures/turkce_notlar.pdf` dosyasının parçaları kullanılıyor; LLM yerine
  `FakeLLM` var, yani ağa veya Ollama'ya hiç gidilmiyor.

## Neden?

- **Model notlarını bilmiyor.** `gemma2:9b` genel bilgiyle eğitildi; senin hocanın slaytını
  görmedi. Ona doğru metni prompt içinde vermezsek ya "bilmiyorum" der ya da (daha kötüsü)
  uydurur.
- **Halüsinasyon tehlikesi.** Dil modelleri akıcı ama yanlış cümleler kurabilir. Öğrenci
  için bu ciddi bir sorun: sınava yanlış bilgiyle girer. Bu yüzden iki kat koruma koyduk:
  1. Eşik altındaki aramalarda **LLM'i hiç çağırmıyoruz**.
  2. LLM'e "sadece kaynaklardan cevap ver, yoksa `found=false` de" diyoruz.
- **Kaynak göstermek güven verir.** Cevabın yanında "s. 4" görmek öğrencinin notuna dönüp
  kontrol etmesini sağlar. Ama LLM'in söylediği numaraya körü körüne güvenemeyiz; bu yüzden
  doğruluyoruz.
- **Maliyet ve hız.** Alakasız bir soruda LLM çağrısı yapmamak yerel makinede saniyeler
  kazandırır.

## Nasıl çalışıyor?

`ask` fonksiyonu RAG'in adımlarını sırayla uygular:

1. **Hazırlık:** soru `strip()` edilir; boşsa `ValueError` fırlatılır.
2. **Retrieve (getir):** `store.search(question, k=k, doc_id=doc_id)` en benzer `k` parçayı
   skor sırasına göre (en yüksek önce) döner. `doc_id` verilirse arama tek bir belgeyle
   sınırlanır; bilinmeyen `doc_id` kontrolü API katmanının işi (Görev 10).
3. **Filtre:** `score >= min_score` olmayan parçalar atılır. Hiçbiri kalmazsa:
   `Answer(answer=NOT_FOUND_MESSAGE, found=False, citations=[])` — **LLM çağrılmadan**.
4. **Augment (zenginleştir):** kalan parçalar `format_sources` ile numaralanır; kullanıcı
   mesajı `build_user_prompt` ile "kaynaklar + `Question: ...`" şeklinde kurulur. Numara
   sırası arama sırasıyla aynıdır, yani [1] her zaman en benzer parçadır.
5. **Generate (üret):** `llm.complete_json(ANSWER_SYSTEM, user, LLMAnswer)` çağrılır. Model
   `{"answer": ..., "found": ..., "cited_sources": [...]}` döner.
6. **Doğrulama:**
   - `found` false ise **veya** cevap boş/sadece boşluksa → `NOT_FOUND_MESSAGE`, kaynak yok.
   - Aksi halde `valid_citations`: `1..len(sources)` aralığı dışındaki numaralar (ör. 0, 9,
     -1) atılır; aynı `(doc_id, filename, page)` iki kez eklenmez; sıra ilk atıf sırasıdır.
     Örnek: model `[3, 1, 3]` derse sonuç `[kaynak 3, kaynak 1]`.
   - Hiç geçerli atıf kalmazsa cevabı yine gösteririz ama `citations=[]` olur.

`ANSWER_SYSTEM` içindeki kurallar özetle:

- Sadece kaynaklardaki bilgiyi kullan, dış bilgi ve tahmin yok.
- Sorunun dilinde cevap ver (Türkçe soru → Türkçe cevap).
- Kullandığın kaynak numaralarını `cited_sources` içine yaz.
- Kaynaklar soruyu kısmen cevaplıyorsa desteklenen kısmı cevapla, eksik olanı söyle.
- Kaynaklarda cevap yoksa `found=false`.

**Bağlam bütçesi:** model 8k token pencereyle çalışıyor. Varsayılan `top_k=6` ve
`chunk_size=1500` karakterle kaynaklar yaklaşık 9000 karakter (~3k token) tutar; sistem
mesajı ve cevap için bolca yer kalır.

## Önemli kavramlar

- **RAG (Retrieval-Augmented Generation):** modeli yeniden eğitmeden, soruyla ilgili
  belgeleri bulup prompt'a ekleyerek cevap ürettirmek. Üç adım: retrieve → augment →
  generate.
- **Grounding (temellendirme):** cevabın verilen kaynaklara dayanması zorunluluğu.
- **Halüsinasyon:** modelin kaynakta olmayan ama inandırıcı görünen bilgi üretmesi.
- **`min_score` eşiği:** kosinüs benzerliği için alt sınır. Çok düşükse alakasız parçalar
  prompt'a girer; çok yüksekse doğru cevaplar "bulunamadı" olur. Değeri Görev 12'deki
  değerlendirmelerle ayarlanacak (`config.py` içinde `min_score = 0.75`).
- **Atıf doğrulama (citation validation):** LLM çıktısı kullanıcı girdisi gibi
  güvenilmezdir; numaraları aralık kontrolünden geçirip tekilleştiririz.
- **Yapılandırılmış çıktı:** `LLMAnswer` şeması sayesinde `found` ve `cited_sources`
  alanlarını kod güvenle okuyabilir.
- **Test ikizleri:** `HashEmbedder` + `FakeLLM` ile testler deterministik ve hızlıdır;
  `FakeLLM.calls` sayesinde prompt'un içeriğini ve LLM'in hiç çağrılmadığını kontrol
  edebiliriz.

## Mülakatta sorulabilecek sorular

1. **RAG nedir, fine-tuning yerine neden tercih edilir?**
   Bilgi prompt'a eklendiği için yeni not yüklemek anında etkili olur, eğitim maliyeti
   yoktur ve cevap kaynağa bağlanabilir.
2. **Halüsinasyonu nasıl azalttınız?**
   Eşik altı aramada LLM'i çağırmamak, "sadece kaynaklar" kuralı, `found=false` kaçış
   yolu, boş cevabı "bulunamadı" saymak.
3. **LLM var olmayan bir kaynak numarası verirse ne olur?**
   `valid_citations` aralık dışını atar; geçerli olan kalmazsa cevap atıfsız gösterilir.
4. **Neden kaynakları skor sırasıyla numaralıyorsunuz?**
   En alakalı parça [1] olur; model genelde baştaki bilgiye daha çok dikkat eder ve
   test edilebilirlik artar.
5. **`min_score` nasıl seçilmeli?**
   Etiketli soru setiyle (cevabı notlarda olan/olmayan) hem bulma oranını hem "yanlış
   bulundu" oranını ölçüp dengelemek gerekir.
6. **8k bağlam penceresi tasarımı nasıl etkiledi?**
   `k` ve parça boyutu, sistem mesajı + kaynaklar + cevap pencereye sığacak şekilde
   seçildi; promptlar kısa tutuldu.
7. **Bu kodu nasıl test ettiniz?**
   Gerçek Chroma deposu + sahte embedder + `FakeLLM`; ağ yok, sonuç deterministik.
