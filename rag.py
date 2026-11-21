# =============================================================================
#  BASİT RAG: KENDİ DÖKÜMANLARINA SORU SOR
#
#  Çalıştırmak için terminalde, bu dosyanın olduğu klasörde:
#       python rag.py
#
#  Akış:  dökümanları oku → parçala → vektöre çevir → (soru gelince)
#         en benzer parçaları bul → parçaları soruyla birlikte modele gönder
# =============================================================================

import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------
# Klasörleri bu dosyanın bulunduğu yere göre belirliyoruz. Böylece terminal
# hangi klasörde açık olursa olsun kod dökümanları doğru yerde arar.
try:
    ANA_KLASOR = Path(__file__).resolve().parent
except NameError:  # Jupyter / etkileşimli pencerede __file__ tanımlı değildir
    ANA_KLASOR = Path.cwd()

DOKUMAN_KLASORU = ANA_KLASOR / "dokumanlar"
INDEKS_KLASORU = ANA_KLASOR / "indeks"

PARCA_KELIME = 250    # Bir parçada kaç kelime olsun
ORTUSME_KELIME = 50   # Ardışık parçalar kaç kelime paylaşsın
K = 4                 # Her soruda kaç parça getirilsin

EMBED_MODEL = "intfloat/multilingual-e5-small"  # Türkçe bilen embedding modeli

# Cevabı kim üretsin?
#   "groq" → GROQ_API_KEY ortam değişkeni gerekir
#   "elle" → anahtar gerekmez; hazır komut prompt.txt'ye yazılır, claude.ai'ye yapıştırırsın
URETICI = "groq"
GROQ_MODEL = "openai/gpt-oss-120b"  # Hata verirse: console.groq.com/docs/models


def dur(mesaj: str):
    """Anlaşılır bir mesaj yazıp programı durdurur (arayüz de bu mesajı yakalar)."""
    raise SystemExit("\n[SORUN] " + mesaj)


# ---------------------------------------------------------------------------
# 1) DÖKÜMANLARI OKUMA
# ---------------------------------------------------------------------------
# Her dosyayı sayfa sayfa okuyup {kaynak, sayfa, metin} kaydına çeviriyoruz.
# Sayfa numarasını tutmamızın nedeni, cevabın kaynağını gösterebilmek.

def dosya_sayfalari(ad: str, veri: bytes) -> list[dict]:
    """Tek bir dosyayı (bayt olarak) sayfa sayfa okur. Hem klasörden okunan
    dosyalar hem de arayüze yüklenen dosyalar bu fonksiyondan geçer."""
    uzanti = Path(ad).suffix.lower()
    if uzanti == ".pdf":
        import io
        from pypdf import PdfReader
        return [{"kaynak": ad, "sayfa": i + 1, "metin": sayfa.extract_text() or ""}
                for i, sayfa in enumerate(PdfReader(io.BytesIO(veri)).pages)]
    if uzanti in {".txt", ".md"}:
        return [{"kaynak": ad, "sayfa": 1, "metin": veri.decode("utf-8", errors="ignore")}]
    return []  # Desteklenmeyen tür


def dokumanlari_oku() -> list[dict]:
    if not DOKUMAN_KLASORU.exists():
        DOKUMAN_KLASORU.mkdir()
        dur(f"'dokumanlar' klasörü yoktu, oluşturdum:\n  {DOKUMAN_KLASORU}\n"
            "İçine PDF/TXT dosyalarını koyup tekrar çalıştır.")
    if not DOKUMAN_KLASORU.is_dir():
        dur(f"'dokumanlar' bir klasör değil, bir dosya:\n  {DOKUMAN_KLASORU}\n"
            "Bu dosyanın adını değiştir (ör. not.txt), 'dokumanlar' adında bir klasör "
            "oluştur ve dosyayı içine taşı.")

    sayfalar = []
    for yol in sorted(DOKUMAN_KLASORU.iterdir()):
        if not yol.is_file():
            continue
        okunan = dosya_sayfalari(yol.name, yol.read_bytes())
        if not okunan:
            print(f"  (atlandı, desteklenmeyen tür: {yol.name})")
        sayfalar.extend(okunan)

    if not sayfalar:
        dur(f"'dokumanlar' klasöründe okunabilen dosya yok:\n  {DOKUMAN_KLASORU}\n"
            "İçine .pdf, .txt veya .md dosyası koy.")

    bos = [f"{s['kaynak']} s.{s['sayfa']}" for s in sayfalar if len(s["metin"].strip()) < 20]
    print(f"{len(sayfalar)} sayfa okundu ({len(bos)} tanesi boş).")
    if bos and len(bos) > len(sayfalar) / 2:
        print("  UYARI: Sayfaların çoğu boş. PDF'ler taranmış (resim) olabilir; "
              "bu durumda önce OCR gerekir.")
    return sayfalar


# ---------------------------------------------------------------------------
# 2) PARÇALAMA
# ---------------------------------------------------------------------------
# Uzun metni örtüşen parçalara bölüyoruz. Örtüşme, iki parçanın sınırına denk
# gelen bir tanımın en az bir parçada bütün kalmasını sağlar.
#   [0 ...... 249]
#            [200 ...... 449]        adım = 250 - 50 = 200
#                       [400 ...... 649]

def parcala(sayfalar: list[dict], yazdir: bool = True) -> list[dict]:
    adim = PARCA_KELIME - ORTUSME_KELIME
    parcalar = []
    for s in sayfalar:
        kelimeler = s["metin"].split()
        for bas in range(0, len(kelimeler), adim):
            parcalar.append({"kaynak": s["kaynak"], "sayfa": s["sayfa"],
                             "metin": " ".join(kelimeler[bas:bas + PARCA_KELIME])})
            if bas + PARCA_KELIME >= len(kelimeler):
                break
    if yazdir:
        print(f"{len(parcalar)} parça oluşturuldu.")
    return parcalar


# ---------------------------------------------------------------------------
# 3) EMBEDDING ve İNDEKS
# ---------------------------------------------------------------------------
# Her parçayı anlamını temsil eden bir vektöre çeviriyoruz. Anlamca benzer
# metinlerin vektörleri birbirine yakın olur. Sonuç: satırları parçalar olan
# bir X matrisi (n parça x p boyut).
# Vektörleri 1 uzunluğa getiriyoruz (normalize). Böylece iki vektörün iç
# çarpımı doğrudan kosinüs benzerliğini verir.
# E5 modeli parçaların başına "passage: ", soruların başına "query: " ister.
#
# Hesap uzun sürdüğü için sonucu 'indeks' klasörüne kaydediyoruz. Dökümanlar
# değişmediyse bir sonraki çalıştırmada hesaplamadan yüklüyoruz.

def vektorlestir(model, parcalar: list[dict], ilerleme: bool = False) -> np.ndarray:
    """Parçaları (n x p) boyutlu, satırları birim uzunlukta bir matrise çevirir."""
    X = model.encode(["passage: " + p["metin"] for p in parcalar],
                     normalize_embeddings=True, show_progress_bar=ilerleme)
    return np.asarray(X, dtype=np.float32)


def dokuman_imzasi() -> list:
    """Dosya adı, boyutu ve değişme zamanı. Biri değişirse indeks yeniden kurulur."""
    return [[p.name, p.stat().st_size, int(p.stat().st_mtime)]
            for p in sorted(DOKUMAN_KLASORU.iterdir()) if p.is_file()] + \
           [PARCA_KELIME, ORTUSME_KELIME, EMBED_MODEL]


def indeksi_hazirla(model):
    imza_dosyasi = INDEKS_KLASORU / "imza.json"
    if imza_dosyasi.exists() and DOKUMAN_KLASORU.is_dir():
        if json.loads(imza_dosyasi.read_text(encoding="utf-8")) == dokuman_imzasi():
            X = np.load(INDEKS_KLASORU / "vektorler.npy")
            parcalar = json.loads((INDEKS_KLASORU / "parcalar.json").read_text(encoding="utf-8"))
            print(f"Kayıtlı indeks yüklendi ({len(parcalar)} parça).")
            return X, parcalar

    parcalar = parcala(dokumanlari_oku())
    print("Parçalar vektöre çevriliyor (ilk seferde biraz sürer)...")
    X = vektorlestir(model, parcalar, ilerleme=True)

    INDEKS_KLASORU.mkdir(exist_ok=True)
    np.save(INDEKS_KLASORU / "vektorler.npy", X)
    (INDEKS_KLASORU / "parcalar.json").write_text(
        json.dumps(parcalar, ensure_ascii=False), encoding="utf-8")
    imza_dosyasi.write_text(json.dumps(dokuman_imzasi()), encoding="utf-8")
    print(f"İndeks kuruldu ve kaydedildi: {X.shape[0]} parça x {X.shape[1]} boyut.")
    return X, parcalar


# ---------------------------------------------------------------------------
# 4) ARAMA
# ---------------------------------------------------------------------------
# Soruyu da vektöre çevirip (q) her parçayla benzerliğini hesaplıyoruz:
#   skor = X @ q      (her parça için kosinüs benzerliği, -1 ile 1 arası)
# En yüksek skorlu K parçayı döndürüyoruz.

def ara(soru, model, X, parcalar, k=K):
    q = model.encode(["query: " + soru], normalize_embeddings=True)[0]
    skorlar = X @ q
    en_iyi = np.argsort(-skorlar)[:k]
    return [{**parcalar[i], "skor": float(skorlar[i])} for i in en_iyi]


# ---------------------------------------------------------------------------
# 5) ÜRETİM
# ---------------------------------------------------------------------------
# Bulunan parçaları numaralandırıp soruyla birlikte modele veriyoruz.
# "Sadece bağlama dayan, yoksa bilmediğini söyle" talimatı uydurmayı azaltır.

SISTEM_TALIMATI = (
    "Sen bir ders asistanısın. Soruyu YALNIZCA verilen bağlam parçalarına "
    "dayanarak Türkçe cevapla. Kullandığın her bilginin sonuna ilgili parçanın "
    "numarasını [1] biçiminde yaz. Bağlamda cevap yoksa bunu açıkça söyle, "
    "tahmin yürütme."
)


def cevapla(soru, bulunanlar, llm):
    baglam = "\n\n".join(f"[{n}] ({p['kaynak']}, s.{p['sayfa']})\n{p['metin']}"
                         for n, p in enumerate(bulunanlar, start=1))
    kullanici_mesaji = f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"

    if URETICI == "elle":
        (ANA_KLASOR / "prompt.txt").write_text(
            SISTEM_TALIMATI + "\n\n" + kullanici_mesaji, encoding="utf-8")
        return "Komut prompt.txt dosyasına yazıldı. İçeriğini claude.ai'ye yapıştır."

    yanit = llm.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,  # Düşük = metne sadık, az yaratıcı
        messages=[{"role": "system", "content": SISTEM_TALIMATI},
                  {"role": "user", "content": kullanici_mesaji}],
    )
    return yanit.choices[0].message.content


def llm_hazirla():
    if URETICI == "elle":
        return None
    anahtar = os.environ.get("GROQ_API_KEY")
    if not anahtar:
        dur("GROQ_API_KEY bulunamadı. Komut İstemi'nde:\n"
            '  setx GROQ_API_KEY "anahtarın"\n'
            "çalıştır, sonra VS Code'u tamamen kapatıp aç.\n"
            "Anahtarsız denemek için dosyada URETICI = \"elle\" yap.")
    from groq import Groq
    return Groq(api_key=anahtar)


# ---------------------------------------------------------------------------
# ANA PROGRAM
# ---------------------------------------------------------------------------
def main():
    llm = llm_hazirla()  # Anahtar sorunu varsa model indirmeden önce söylesin

    print("Embedding modeli yükleniyor (ilk seferde ~470 MB indirilir)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    X, parcalar = indeksi_hazirla(model)

    print("\nHazır. Sorunu yaz (çıkmak için boş bırakıp Enter).")
    while True:
        soru = input("\nSORU: ").strip()
        if not soru:
            break
        bulunanlar = ara(soru, model, X, parcalar)

        print("\nBulunan parçalar:")
        for n, p in enumerate(bulunanlar, start=1):
            print(f"  [{n}] {p['kaynak']} s.{p['sayfa']} (benzerlik {p['skor']:.2f}): "
                  f"{p['metin'][:70]}...")

        try:
            print("\nCEVAP:\n" + cevapla(soru, bulunanlar, llm))
        except Exception as hata:
            print(f"\n[SORUN] Modelden cevap alınamadı: {hata}\n"
                  "Model adı eskimiş olabilir (GROQ_MODEL) ya da anahtar geçersiz olabilir.")


if __name__ == "__main__":
    main()
