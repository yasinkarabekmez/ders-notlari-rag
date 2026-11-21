# Ders Notları için RAG Asistanı

Notlarını (PDF/TXT/MD) yükleyip soru sorduğun, cevapların **yalnızca yüklediğin notlara dayandığı ve kaynak sayfasıyla geldiği** küçük bir RAG (Retrieval-Augmented Generation) uygulaması. Zaman Serileri Analizi ders notlarıyla geliştirildi, ama herhangi bir döküman setiyle çalışır.

LangChain veya vektör veritabanı gibi hazır çerçeveler kullanılmadı. Parçalama, embedding ve benzerlik araması `numpy` ile elle yazıldı. Amaç, RAG'in her adımını görünür kılmak.

## Nasıl çalışır?

```
yüklenen dosyalar ──► sayfa sayfa oku ──► örtüşen parçalara böl ──► her parçayı vektöre çevir (X)
                                                                                     │
soru ──► vektöre çevir (q) ──► kosinüs benzerliği (X @ q) ──► en yakın K parça ──► LLM ──► kaynaklı cevap
```

1. **Okuma:** PDF'ler sayfa sayfa okunur. Sayfa numarası, cevapta kaynak göstermek için saklanır.
2. **Parçalama:** Metin 250 kelimelik, 50 kelime örtüşen parçalara bölünür. Örtüşme, iki parçanın sınırına denk gelen bir tanımın bölünmemesini sağlar.
3. **Embedding:** Her parça [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) modeliyle vektöre çevrilir. Model Türkçe biliyor ve sunucuda yerel olarak çalışıyor. Vektörler birim uzunluğa getirildiği için iç çarpım doğrudan kosinüs benzerliğini verir.
4. **Arama:** Soru da vektöre çevrilir ve en benzer K parça seçilir.
5. **Üretim:** Bulunan parçalar, "yalnızca bu bağlama dayan, yoksa bilmediğini söyle" talimatıyla birlikte dil modeline (Groq üzerinden) gönderilir.

Her kullanıcının notları kendi oturumunda işlenir ve diske kaydedilmez. Aynı dosyalar tekrar yüklenirse indeks önbellekten gelir.

## Kendi bilgisayarında çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda açılan sayfada soldaki **Notların** alanına dosyalarını yükle ve soru sor. İlk açılışta embedding modeli indirilir (~470 MB).

Cevapları üreten model için [console.groq.com](https://console.groq.com/keys) adresinden ücretsiz bir API anahtarı gerekir. Anahtarı ortam değişkeni olarak tanımlayabilirsin, tanımlamazsan arayüz kenar çubuğunda ister:

```bash
# Windows (sonra terminali/VS Code'u yeniden aç)
setx GROQ_API_KEY "anahtarın"

# macOS / Linux
export GROQ_API_KEY="anahtarın"
```

**Terminal sürümü:** `dokumanlar/` klasörüne dosya koyup `python rag.py` ile arayüz olmadan da kullanabilirsin.

## İnternette yayınlama (Streamlit Community Cloud, ücretsiz)

1. Bu depoyu GitHub'a yükle.
2. [share.streamlit.io](https://share.streamlit.io) adresine GitHub hesabınla gir ve **Create app** de.
3. Depoyu seç, **Main file path** olarak `app.py` yaz.
4. **Advanced settings → Secrets** alanına anahtarını ekle:
   ```toml
   GROQ_API_KEY = "anahtarın"
   ```
   Bunu yaparsan kullanıcılar senin anahtarını kullanır, kendi anahtarlarını girmeleri gerekmez. Eklemezsen her kullanıcıdan kendi anahtarı istenir.
5. **Deploy** de. İlk kurulum birkaç dakika sürer. Sonunda `https://...streamlit.app` biçiminde paylaşılabilir bir bağlantı alırsın.

> Anahtarı secrets'a eklersen uygulamayı kullanan herkesin soruları senin Groq kotandan düşer. Bağlantıyı kimlerle paylaştığına dikkat et.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `app.py` | Streamlit arayüzü: dosya yükleme, oturuma özel indeks, sohbet. |
| `rag.py` | RAG'in kendisi: okuma, parçalama, embedding, arama, cevap üretme. Tek başına terminalde de çalışır. |
| `.streamlit/config.toml` | Arayüzün renk teması ve yükleme sınırı. |
| `requirements.txt` | Kütüphaneler (bulutta sığması için PyTorch'un CPU sürümü). |
| `dokumanlar/` | Terminal sürümü için döküman klasörü (örnek bir not içerir). |

## Ayarlar

`rag.py` dosyasının başında:

| Ayar | Varsayılan | Anlamı |
|---|---|---|
| `PARCA_KELIME` | 250 | Bir parçadaki kelime sayısı |
| `ORTUSME_KELIME` | 50 | Ardışık parçaların paylaştığı kelime sayısı |
| `K` | 4 | Her soruda getirilen parça sayısı (arayüzden de değiştirilebilir) |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Embedding modeli |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Cevabı yazan model ([güncel liste](https://console.groq.com/docs/models)) |

`app.py` içinde `MAKS_DOSYA` (10) ve `MAKS_SAYFA` (600) yükleme sınırlarını belirler.

## Bilinen sınırlamalar

- PDF'lerden yalnızca metin çıkarılır. Formüller, tablolar ve şekiller bozulabilir ya da kaybolabilir. Taranmış (resim) PDF'ler okunamaz.
- Arama her zaman en yakın K parçayı getirir. Notlarda karşılığı olmayan bir soruda modelin "bağlamda yok" demesi talimata bağlıdır.
- Yüklenen notların ilgili parçaları cevap üretmek için Groq'a gönderilir.
- Cevaplar modelin yorumudur. Kaynak sayfasından kontrol edilmelidir.
