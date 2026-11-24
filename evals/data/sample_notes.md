<!-- Self-written sample lecture notes for the evaluation set. Original text, MIT License, same as this repository. -->
Makine Öğrenmesine Giriş — Ders Notları
Bölüm 1: Tanım ve Kısa Tarihçe

Makine öğrenmesi, bilgisayarların her adımı açıkça programlanmadan, veriden örüntü çıkararak bir görevdeki başarımını deneyimle artırmasını inceleyen yapay zekâ alt alanıdır. Tom Mitchell'in sık alıntılanan tanımına göre bir program, bir T görevinde P ölçütüyle ölçülen başarımı E deneyimiyle iyileşiyorsa o görevi öğreniyor sayılır. Örneğin e-posta süzgecinde görev istenmeyen iletileri ayırmak, deneyim etiketlenmiş eski e-postalar, ölçüt ise doğru sınıflandırılan iletilerin oranıdır.

Geleneksel programlamada geliştirici kuralları yazar, bilgisayar bu kuralları veriye uygular ve sonucu üretir. Makine öğrenmesinde ise veri ile beklenen sonuçlar birlikte verilir; kuralları, yani modeli, öğrenme algoritması kendisi bulur. Bu yaklaşım, yüz tanıma ya da konuşmayı yazıya dökme gibi kuralları elle yazmanın neredeyse imkânsız olduğu problemlerde özellikle değerlidir.

Alanın kökleri 1950'lere uzanır. Arthur Samuel 1959'da kendi kendine oynayarak gelişen bir dama programı geliştirmiş ve "makine öğrenmesi" ifadesini yaygınlaştırmıştır. Frank Rosenblatt 1958'de algılayıcıyı (perceptron) tanıtmış, ancak 1969'da Minsky ve Papert tek katmanlı algılayıcının XOR gibi doğrusal olarak ayrılamayan problemleri çözemeyeceğini göstermiştir. Bu eleştiri ve beklentilerin karşılanamaması, araştırma fonlarının azaldığı "yapay zekâ kışı" dönemlerine katkıda bulunmuştur.

1986'da geri yayılım algoritmasının yeniden popüler hâle gelmesiyle çok katmanlı ağlar eğitilebilir oldu. 1990'larda destek vektör makineleri ve topluluk yöntemleri öne çıktı. 2012'de AlexNet adlı derin evrişimli ağın ImageNet yarışmasını büyük farkla kazanması, derin öğrenme çağını başlattı. Bu sıçramanın üç temel nedeni büyük etiketli veri kümeleri, grafik işlemcilerin (GPU) sağladığı hesaplama gücü ve daha iyi eğitim teknikleridir.

Makine öğrenmesi genellikle üç ana gruba ayrılır: denetimli öğrenme, denetimsiz öğrenme ve pekiştirmeli öğrenme. Pekiştirmeli öğrenmede bir ajan çevreyle etkileşime girer, eylemlerinin sonucunda ödül ya da ceza alır ve toplam ödülü en üst düzeye çıkaran bir politika öğrenir. Bu derste ağırlıklı olarak ilk iki gruba odaklanacağız.
---page---
Bölüm 2: Denetimli Öğrenme

Denetimli öğrenmede eğitim verisi girdi–çıktı çiftlerinden oluşur. Her örnek için doğru cevap, yani etiket, bilinir. Amaç, girdiden çıktıya giden ve daha önce görülmemiş örnekler için de iyi tahmin yapan bir fonksiyon öğrenmektir. Etiketler çoğunlukla insanlar tarafından hazırlandığı için bu tür veriyi toplamak zaman alıcı ve pahalı olabilir.

Denetimli öğrenme problemleri iki türe ayrılır. Sınıflandırmada çıktı ayrık bir kategoridir: bir tümörün iyi huylu ya da kötü huylu olması veya bir resimde kedi mi köpek mi bulunduğu gibi. Regresyonda ise çıktı sürekli bir sayıdır: bir evin metrekaresine ve konumuna bakarak fiyatını tahmin etmek ya da yarınki sıcaklığı öngörmek gibi.

En basit regresyon modeli doğrusal regresyondur. Model, girdilerin ağırlıklı toplamına bir sabit terim ekleyerek tahmin üretir. Ağırlıklar, tahminler ile gerçek değerler arasındaki farkların karelerinin ortalaması (ortalama kare hata) en küçük olacak şekilde seçilir. Bu kayıp fonksiyonu büyük hataları daha fazla cezalandırır.

İsmine rağmen lojistik regresyon bir sınıflandırma yöntemidir. Ağırlıklı toplamı sigmoid fonksiyonundan geçirerek 0 ile 1 arasında bir olasılık üretir; olasılık belirlenen eşiğin, örneğin 0,5'in üzerindeyse örnek pozitif sınıfa atanır. Eğitimde genellikle çapraz entropi kaybı kullanılır.

Ağırlıkları bulmak için en yaygın yol gradyan inişidir. Kaybın her ağırlığa göre türevi hesaplanır ve ağırlıklar kaybı azaltan yönde küçük adımlarla güncellenir. Adım büyüklüğünü öğrenme oranı belirler: çok büyük seçilirse kayıp salınım yapar veya ıraksar, çok küçük seçilirse eğitim gereğinden uzun sürer. Verinin tamamı yerine küçük rastgele parçalarla (mini-batch) yapılan güncellemeye stokastik gradyan inişi denir.

k-en yakın komşu algoritması ise neredeyse hiç eğitim yapmaz. Yeni bir örneği sınıflandırırken eğitim kümesindeki en yakın k örneğe bakar ve çoğunluğun sınıfını seçer. Basit olmasına karşın büyük veri kümelerinde tahmin aşaması yavaştır ve özelliklerin ölçeklenmesine duyarlıdır; bu yüzden özellikler genellikle önce aynı aralığa getirilir.
---page---
Bölüm 3: Denetimsiz Öğrenme ve Kümeleme

Denetimsiz öğrenmede verinin etiketleri yoktur. Algoritmanın görevi, verinin içindeki yapıyı kendi başına keşfetmektir: benzer örnekleri gruplamak, verinin boyutunu azaltmak ya da olağandışı örnekleri bulmak gibi. Etiket gerektirmediği için çok büyük miktarda ham veriye uygulanabilir.

En bilinen kümeleme yöntemi k-ortalamalar (k-means) algoritmasıdır. Önce küme sayısı k seçilir ve k merkez rastgele yerleştirilir. Ardından iki adım dönüşümlü olarak tekrarlanır: her nokta en yakın merkeze atanır, sonra her merkez kendisine atanan noktaların ortalamasına taşınır. Atamalar değişmeyene kadar devam edilir. Algoritma hızlıdır, fakat başlangıç merkezlerine duyarlıdır ve kümelerin yaklaşık küresel olduğunu varsayar; bu nedenle farklı başlangıçlarla birkaç kez çalıştırılması önerilir.

k değerini seçmek için dirsek yöntemi kullanılabilir. Farklı k değerleri için küme içi kareler toplamı çizilir; eğrinin belirgin biçimde kırıldığı, yani ek kümelerin hatayı artık pek azaltmadığı nokta uygun k olarak alınır. Siluet skoru da bir noktanın kendi kümesine ne kadar yakın, komşu kümeye ne kadar uzak olduğunu ölçerek kümelemenin kalitesini değerlendirir.

Hiyerarşik kümeleme, küme sayısının önceden bilinmediği durumlarda kullanışlıdır. Birleştirici yaklaşımda her nokta tek başına bir küme olarak başlar ve en yakın iki küme adım adım birleştirilir. Sonuç, dendrogram adı verilen ağaç biçiminde bir şemayla gösterilir; ağacın istenen yükseklikte kesilmesiyle küme sayısı belirlenir. DBSCAN ise yoğunluğa dayalıdır: yoğun bölgeleri küme sayar, seyrek bölgelerdeki noktaları gürültü olarak işaretler ve keyfi şekilli kümeleri bulabilir.

Boyut indirgeme için en yaygın teknik temel bileşen analizidir (PCA). PCA, verideki varyansı en çok açıklayan yeni eksenleri bulur ve veriyi bu eksenlerin ilk birkaçına izdüşürür. Böylece yüzlerce özellik birkaç bileşene indirilerek görselleştirme kolaylaşır ve gürültü azalır.

Denetimsiz öğrenmenin gerçek hayattaki kullanımları arasında müşteri segmentasyonu, benzer haberlerin gruplanması ve kredi kartı işlemlerinde dolandırıcılık gibi anomali tespiti sayılabilir.
---page---
Bölüm 4: Aşırı Öğrenme ve Düzenlileştirme

Bir modelin asıl amacı eğitim verisini ezberlemek değil, yeni veriye genelleme yapmaktır. Aşırı öğrenme (overfitting), modelin eğitim verisindeki gürültüyü ve rastlantısal ayrıntıları da öğrenmesi sonucu eğitim hatasının çok düşük, test hatasının ise yüksek olması durumudur. Eksik öğrenmede (underfitting) ise model verideki temel ilişkiyi bile yakalayamayacak kadar basittir; hem eğitim hem test hatası yüksektir.

Bu iki uç, yanlılık–varyans ödünleşimi ile açıklanır. Yanlılık, modelin yanlış veya fazla basit varsayımlarından kaynaklanan sistematik hatadır. Varyans ise modelin eğitim kümesindeki küçük değişikliklere ne kadar duyarlı olduğunu gösterir. Model karmaşıklığı arttıkça yanlılık azalır ama varyans artar; amaç toplam hatayı en küçük yapan dengeyi bulmaktır.

Aşırı öğrenmeyi fark etmek için veri üç parçaya ayrılır: modelin eğitildiği eğitim kümesi, hiperparametrelerin seçildiği doğrulama kümesi ve yalnızca en sonda bir kez kullanılan test kümesi. Veri azsa k-katlı çapraz doğrulama uygulanır: veri k eşit parçaya bölünür, her turda bir parça doğrulama için ayrılır, kalanlarla model eğitilir ve k sonucun ortalaması alınır.

Düzenlileştirme (regularization), kayıp fonksiyonuna büyük ağırlıkları cezalandıran bir terim ekleyerek modeli daha basit kalmaya zorlar. L2 düzenlileştirmesi (ridge) ağırlıkların karelerinin toplamını ekler ve ağırlıkları sıfıra doğru küçültür. L1 düzenlileştirmesi (lasso) ağırlıkların mutlak değerlerinin toplamını ekler; bazı ağırlıkları tam olarak sıfırladığı için aynı zamanda özellik seçimi yapar. Ceza teriminin gücünü lambda adlı hiperparametre belirler.

Sinir ağlarında başka teknikler de kullanılır. Erken durdurmada doğrulama hatası artmaya başladığı anda eğitim kesilir. Dropout yönteminde eğitim sırasında her adımda nöronların bir kısmı rastgele devre dışı bırakılır, böylece ağ tek tek nöronlara aşırı bağımlı hâle gelmez.

En etkili çözümlerden biri de daha fazla veri toplamaktır. Veri toplamak mümkün değilse veri artırma ile mevcut örneklerden yeni örnekler üretilir; örneğin görüntüler döndürülür, kırpılır veya aynalanır.
---page---
Bölüm 5: Değerlendirme Metrikleri

Bir sınıflandırıcının başarısını ölçerken ilk akla gelen metrik doğruluktur (accuracy): doğru tahminlerin tüm tahminlere oranı. Ancak sınıflar dengesiz olduğunda doğruluk yanıltıcıdır. Hastaların yalnızca yüzde birinin hasta olduğu bir veri kümesinde herkese "sağlıklı" diyen bir model yüzde 99 doğruluk elde eder ama tek bir hastayı bile bulamaz.

Bu yüzden karışıklık matrisi kullanılır. İkili sınıflandırmada dört hücre vardır: doğru pozitif (hasta olana hasta demek), yanlış pozitif (sağlıklıya hasta demek), doğru negatif ve yanlış negatif (hastayı gözden kaçırmak). Diğer metrikler bu dört sayıdan hesaplanır.

Kesinlik (precision), modelin pozitif dediği örneklerin ne kadarının gerçekten pozitif olduğunu gösterir: DP / (DP + YP). Duyarlılık (recall), gerçek pozitiflerin ne kadarının yakalandığını gösterir: DP / (DP + YN). Hangisinin önemli olduğu probleme bağlıdır. Spam süzgecinde önemli bir e-postanın yanlışlıkla silinmesi istenmez, bu yüzden kesinlik ön plandadır. Kanser taramasında ise bir hastayı kaçırmak çok daha pahalıdır, bu yüzden duyarlılık ön plandadır.

F1 skoru, kesinlik ile duyarlılığın harmonik ortalamasıdır: 2 × K × D / (K + D). Harmonik ortalama küçük değere daha çok ağırlık verdiği için iki metrikten biri düşükse F1 de düşük çıkar.

Çoğu sınıflandırıcı bir olasılık üretir ve bir eşik değeriyle karar verir. Eşik düşürülürse duyarlılık artar ama kesinlik azalır. ROC eğrisi, tüm eşikler için doğru pozitif oranını yanlış pozitif oranına karşı çizer. Eğrinin altında kalan alan (AUC) 0,5 ise model rastgele tahminden iyi değildir; 1'e yaklaştıkça model pozitifleri negatiflerden daha iyi ayırır.

Regresyon problemlerinde farklı metrikler kullanılır. Ortalama mutlak hata (MAE) hataların mutlak değerlerinin ortalamasıdır ve aykırı değerlere daha dayanıklıdır. Kök ortalama kare hata (RMSE) büyük hataları daha fazla cezalandırır ve hedefle aynı birimdedir. Belirlilik katsayısı R², modelin hedefteki değişkenliğin ne kadarını açıkladığını gösterir.

Son olarak, metrikler her zaman modelin eğitimde hiç görmediği test verisi üzerinde hesaplanmalıdır.
---page---
Bölüm 6: Karar Ağaçları ve Rastgele Orman

Karar ağacı, veriyi bir dizi evet–hayır sorusuyla bölerek tahmin yapan bir modeldir. Kökten başlayan her iç düğüm bir özelliği test eder ("yaş 30'dan büyük mü?"), dallar testin sonucunu, yapraklar ise tahmin edilen sınıfı veya değeri temsil eder. Ağaçlar insanlar tarafından kolayca okunabildiği için yorumlanabilirliği yüksek modellerdir.

Ağaç oluşturulurken her düğümde, veriyi en saf alt gruplara ayıran özellik ve eşik seçilir. Saflığı ölçmek için iki yaygın ölçü vardır. Gini safsızlığı, rastgele seçilen bir örneğin yanlış etiketlenme olasılığını ölçer. Entropi ise bir düğümdeki belirsizliği ölçer; bir bölünmeden önceki ve sonraki entropi arasındaki farka bilgi kazancı denir. Algoritma bilgi kazancını en büyük yapan bölünmeyi açgözlü biçimde seçer.

Derinliği sınırlanmayan bir ağaç, her yaprakta tek bir örnek kalana kadar büyüyebilir ve eğitim verisini ezberler. Bunu önlemek için en büyük derinlik veya bir yapraktaki en az örnek sayısı gibi sınırlar konur ya da ağaç büyütüldükten sonra budama yapılarak gereksiz dallar kesilir.

Tek bir ağaç kararsızdır: veride küçük bir değişiklik bile bambaşka bir ağaç üretebilir. Topluluk (ensemble) yöntemleri bu sorunu birçok modelin tahminini birleştirerek çözer. Torbalama (bagging) yönteminde, eğitim kümesinden yerine koyarak örnekleme ile çok sayıda bootstrap örneklemi oluşturulur ve her biri üzerinde ayrı bir model eğitilir.

Rastgele orman, torbalamayı karar ağaçlarına uygular ve bir rastgelelik daha ekler: her bölünmede tüm özellikler yerine yalnızca rastgele seçilmiş bir özellik alt kümesi değerlendirilir. Bu, ağaçların birbirine benzemesini engeller. Sınıflandırmada ağaçların oy çokluğu, regresyonda ortalaması alınır. Sonuçta tek bir ağaca göre varyansı çok daha düşük bir model elde edilir. Rastgele orman ayrıca hangi özelliklerin tahmine ne kadar katkı yaptığını gösteren özellik önemi değerleri verir.

Gradyan artırma (boosting) ise ağaçları sırayla eğitir: her yeni ağaç, önceki ağaçların toplamının yaptığı hataları düzeltmeye çalışır. XGBoost ve LightGBM bu fikrin popüler uygulamalarıdır ve tablo biçimindeki verilerde çoğu zaman en iyi sonuçları verir.
---page---
Bölüm 7: Yapay Sinir Ağları

Yapay sinir ağları, birbirine bağlı basit hesaplama birimlerinden, yani yapay nöronlardan oluşur. Her nöron girdilerini ağırlıklarla çarpar, bunlara bir sapma (bias) terimi ekler ve sonucu bir aktivasyon fonksiyonundan geçirir. Nöronlar katmanlar hâlinde dizilir: girdi katmanı, bir veya daha fazla gizli katman ve çıktı katmanı. Birden çok gizli katmana sahip ağlara derin ağ denir.

Aktivasyon fonksiyonları ağa doğrusal olmayan bir yapı kazandırır. Bunlar olmasaydı, katman sayısı ne olursa olsun ağ tek bir doğrusal dönüşüme eşdeğer olurdu. Sigmoid çıktıyı 0 ile 1 arasına sıkıştırır, tanh ise -1 ile 1 arasına. Günümüzde gizli katmanlarda en çok ReLU kullanılır: negatif girdiler için 0, pozitif girdiler için girdinin kendisini verir. ReLU hesaplaması ucuzdur ve derin ağlarda gradyanın kaybolması sorununu hafifletir. Çok sınıflı problemlerde çıktı katmanında softmax, puanları toplamı 1 olan olasılıklara dönüştürür.

Eğitim iki aşamadan oluşur. İleri yayılımda girdi katman katman işlenerek bir tahmin üretilir ve kayıp hesaplanır. Geri yayılımda zincir kuralı kullanılarak kaybın her ağırlığa göre gradyanı çıktıdan girdiye doğru hesaplanır; ardından ağırlıklar bir optimizasyon algoritmasıyla güncellenir. Adam gibi uyarlamalı optimizasyon yöntemleri her ağırlık için öğrenme oranını ayrı ayarlar. Eğitim verisinin bütünüyle bir kez işlenmesine bir epok denir.

Çok derin ağlarda gradyanlar katmanlar boyunca çarpıldığı için giderek küçülebilir (kaybolan gradyan) ya da aşırı büyüyebilir (patlayan gradyan). Uygun ağırlık başlatma, ReLU ve toplu normalleştirme (batch normalization) bu sorunları azaltır.

Farklı veri türleri için özel mimariler geliştirilmiştir. Evrişimli sinir ağları (CNN), görüntüdeki kenar ve doku gibi yerel örüntüleri küçük filtrelerle yakalar ve aynı filtreyi görüntünün her yerinde paylaşarak parametre sayısını azaltır. Tekrarlayan sinir ağları (RNN) ve LSTM, metin veya zaman serisi gibi sıralı verileri işler. Transformer mimarisi ise dikkat mekanizması sayesinde dizinin tüm öğeleri arasındaki ilişkileri paralel olarak hesaplar ve günümüzün büyük dil modellerinin temelini oluşturur.
---page---
Bölüm 8: Etik, Önyargı ve Sorumlu Yapay Zekâ

Makine öğrenmesi modelleri işe alım, kredi onayı ve tıbbi teşhis gibi insanların hayatını doğrudan etkileyen kararlarda kullanılıyor. Bu nedenle bir modelin yalnızca doğru değil, aynı zamanda adil, şeffaf ve hesap verebilir olması gerekir.

Modeller önyargıyı çoğunlukla eğitim verisinden devralır. Geçmişte belirli bir gruba ayrımcılık yapılmışsa, bu kararlarla eğitilen model aynı ayrımcılığı sürdürür; buna tarihsel önyargı denir. Örneklem önyargısı, verinin gerçek nüfusu temsil etmemesinden doğar: ağırlıklı olarak açık tenli yüzlerle eğitilmiş bir yüz tanıma sistemi, koyu tenli kişilerde çok daha fazla hata yapabilir. Ölçüm önyargısında ise kullanılan etiket hedeflenen kavramı doğru yansıtmaz; örneğin "tutuklanma" verisi gerçek suç oranının yerine kullanıldığında bazı mahalleler sistematik olarak daha riskli görünür.

Hassas bir özelliği, örneğin cinsiyeti veya etnik kökeni veriden çıkarmak tek başına yeterli değildir. Posta kodu ya da okunan okul gibi başka özellikler bu bilgiyle güçlü biçimde ilişkili olabilir ve vekil değişken olarak aynı ayrımcılığa yol açabilir.

Adaleti ölçmek için çeşitli tanımlar önerilmiştir. Demografik eşitlik, olumlu kararların gruplar arasında benzer oranlarda verilmesini ister. Fırsat eşitliği ise gerçekten nitelikli kişilerin hangi gruptan olurlarsa olsunlar aynı oranda doğru tanınmasını, yani grupların doğru pozitif oranlarının eşit olmasını ister. Bu tanımların hepsini aynı anda sağlamak çoğu zaman matematiksel olarak mümkün değildir; hangi tanımın seçileceği bir değer tercihidir.

Açıklanabilirlik de önemlidir. Kredi başvurusu reddedilen bir kişi, kararın nedenini öğrenme hakkına sahip olmalıdır. Bu yüzden karmaşık modellerin kararlarını yorumlamak için hangi özelliğin tahmini ne kadar etkilediğini gösteren yöntemler kullanılır.

Gizlilik açısından eğitim verisi kişisel bilgiler içerebilir ve Türkiye'de KVKK, Avrupa Birliği'nde GDPR gibi düzenlemelere uyulmalıdır. Sorumlu bir geliştirme süreci; verinin kaynağını belgelemeyi, modeli farklı gruplar üzerinde ayrı ayrı test etmeyi ve yüksek riskli kararlarda son sözü bir insana bırakmayı içerir.
