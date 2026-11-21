# =============================================================================
#  RAG SOHBET ARAYÜZÜ (Streamlit)
#
#  Çalıştırmak için terminalde, bu dosyanın olduğu klasörde:
#       streamlit run app.py
#  Tarayıcıda otomatik açılır (açılmazsa: http://localhost:8501).
#  Kapatmak için terminalde Ctrl+C.
#
#  Kullanıcı notlarını arayüzden yükler, sistem onları o anda işler ve
#  kullanıcı bu notlara soru sorar. Her kullanıcının notları kendi oturumunda
#  kalır; diske kaydedilmez. Arama/cevap mantığı rag.py'de.
#
#  API anahtarı şu sırayla aranır:
#    1) Streamlit secrets (bulutta yayınlarken: GROQ_API_KEY)
#    2) GROQ_API_KEY ortam değişkeni (kendi bilgisayarında)
#    3) Hiçbiri yoksa kullanıcıdan kenar çubuğunda istenir
#
#  Tasarım (ui-ux-pro-max tasarım sistemi):
#    - Stil: Minimalizm / Swiss — sade, bol boşluk, yüksek kontrast
#    - Yazı: başlıklar Crimson Pro (akademik serif), metin Atkinson Hyperlegible
#      (okunabilirlik için tasarlanmış; uzun cevaplar ve formüller için)
#    - Renk: odak mavisi #2563EB, metin #0F172A, ikincil metin #475569
#    - İkonlar emoji değil, Material ikonları
# =============================================================================

import os
import re

import streamlit as st

import rag  # Aynı klasördeki rag.py

st.set_page_config(page_title="Ders Asistanı", page_icon=":material/school:",
                   layout="centered")

# -----------------------------------------------------------------------------
# GÖRSEL STİL
# Streamlit'in kendi teması .streamlit/config.toml'da (renkler). Yazı tiplerini
# ve kartların görünümünü burada CSS ile ekliyoruz.
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@500;600;700&display=swap');

:root {
  --color-primary: #2563EB;
  --color-foreground: #0F172A;
  --color-muted-foreground: #475569;
  --color-card: #FFFFFF;
  --color-border: #E2E8F0;
  --color-muted: #F1F5FD;
}

/* Metin: okunabilir yazı tipi, 16px taban, 1.6 satır aralığı */
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea {
  font-family: 'Atkinson Hyperlegible', system-ui, sans-serif;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  font-size: 1rem;
  line-height: 1.6;
}

/* Başlıklar: akademik serif */
h1, h2, h3, .da-baslik {
  font-family: 'Crimson Pro', Georgia, serif !important;
  color: var(--color-foreground);
  letter-spacing: -0.01em;
}

/* Üst bilgi alanı */
[data-testid="stMarkdownContainer"] p.da-baslik {
  font-size: 2.4rem; font-weight: 600; line-height: 1.1; margin: 0;
}
.da-alt { color: var(--color-muted-foreground); margin: .35rem 0 1rem 0; font-size: 1rem; }
.da-etiketler { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1.25rem; }
.da-etiket {
  background: var(--color-muted); color: var(--color-foreground);
  border: 1px solid var(--color-border); border-radius: 999px;
  padding: .2rem .75rem; font-size: .875rem; white-space: nowrap;
}

/* Sohbet balonları: beyaz kart, ince kenar */
[data-testid="stChatMessage"] {
  background: var(--color-card);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: .75rem;
}

/* Kaynak kartları */
.da-kaynak {
  border-left: 3px solid var(--color-primary);
  background: var(--color-muted);
  border-radius: 0 8px 8px 0;
  padding: .6rem .8rem; margin: .5rem 0;
}
.da-kaynak-baslik { font-weight: 700; font-size: .9rem; color: var(--color-foreground); }
.da-kaynak-skor { color: var(--color-muted-foreground); font-weight: 400; }
.da-kaynak-metin { color: var(--color-muted-foreground); font-size: .875rem;
                   line-height: 1.5; margin-top: .3rem; }

/* Örnek soru butonları: rahat tıklama alanı (en az 44px) */
.stButton > button { min-height: 44px; border-radius: 10px; text-align: left;
                     transition: border-color .2s ease, background-color .2s ease; }

/* Klavye odağı her zaman görünsün */
button:focus-visible, textarea:focus-visible, input:focus-visible {
  outline: 2px solid var(--color-primary) !important; outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Streamlit'in çalışma mantığı (Shiny'ye benzer):
# Her etkileşimde (dosya yükleme, soru gönderme) bu dosya BAŞTAN SONA yeniden
# çalışır. Bu yüzden:
#   - Embedding modeli @st.cache_resource ile bir kez yüklenir ve TÜM
#     kullanıcılar arasında paylaşılır (herkes için aynı model).
#   - Notlardan kurulan indeks @st.cache_data ile dosyaların içeriğine göre
#     saklanır: aynı dosyalar tekrar işlenmez, farklı dosyalar ayrı indeks alır.
#   - Sohbet geçmişi st.session_state'te, yani her kullanıcının kendi
#     oturumunda tutulur.
# -----------------------------------------------------------------------------

MAKS_SAYFA = 600   # Toplam sayfa sınırı: büyük yüklemeler sunucuyu kilitlemesin
MAKS_DOSYA = 10


@st.cache_resource(show_spinner="Dil modeli hazırlanıyor (ilk açılışta birkaç dakika sürebilir)...")
def embedding_modeli():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(rag.EMBED_MODEL)


@st.cache_data(show_spinner=False, max_entries=20)
def indeks_kur(dosyalar: tuple) -> dict:
    """dosyalar: ((ad, bayt), ...). Sonuç dosya içeriğine göre önbelleğe alınır."""
    sayfalar = []
    for ad, veri in dosyalar:
        sayfalar.extend(rag.dosya_sayfalari(ad, veri))
    if len(sayfalar) > MAKS_SAYFA:
        return {"hata": f"Toplam {len(sayfalar)} sayfa yüklendi; sınır {MAKS_SAYFA} sayfa. "
                        "Daha az dosyayla dene."}
    bos = sum(1 for s in sayfalar if len(s["metin"].strip()) < 20)
    parcalar = rag.parcala(sayfalar, yazdir=False)
    if not parcalar:
        return {"hata": "Yüklenen dosyalardan metin çıkarılamadı. PDF'ler taranmış "
                        "(resim) olabilir; bu sürüm yalnızca metin içeren PDF'leri okuyabiliyor."}
    X = rag.vektorlestir(embedding_modeli(), parcalar)
    return {"X": X, "parcalar": parcalar, "sayfa": len(sayfalar), "bos": bos,
            "dosya": [ad for ad, _ in dosyalar]}


def api_anahtari() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:  # secrets dosyası hiç yoksa Streamlit hata fırlatır
        pass
    return os.environ.get("GROQ_API_KEY")


@st.cache_resource
def groq_istemcisi(anahtar: str):
    from groq import Groq
    return Groq(api_key=anahtar)


def latex_duzelt(metin: str) -> str:
    """Model formülleri \\( ... \\) ve \\[ ... \\] ile yazıyor; Streamlit ise
    $ ... $ ve $$ ... $$ bekliyor. Sadece işaretleri değiştiriyoruz."""
    metin = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", metin, flags=re.S)
    metin = re.sub(r"\\\((.+?)\\\)", r"$\1$", metin, flags=re.S)
    return metin


def html_kacis(metin: str) -> str:
    """PDF metnini HTML içine güvenle koymak için özel karakterleri kaçırır."""
    return metin.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def kaynaklari_goster(bulunanlar):
    with st.expander(f"Kaynaklar ({len(bulunanlar)} parça)", icon=":material/menu_book:"):
        for n, p in enumerate(bulunanlar, start=1):
            ozet = p["metin"][:500] + ("..." if len(p["metin"]) > 500 else "")
            st.markdown(
                f'<div class="da-kaynak">'
                f'<div class="da-kaynak-baslik">[{n}] {html_kacis(p["kaynak"])}, sayfa {p["sayfa"]} '
                f'<span class="da-kaynak-skor">· benzerlik {p["skor"]:.2f}</span></div>'
                f'<div class="da-kaynak-metin">{html_kacis(ozet)}</div></div>',
                unsafe_allow_html=True)


AVATAR = {"user": ":material/person:", "assistant": ":material/school:"}

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []  # Her eleman: {"rol", "metin", "kaynaklar"}

# --- Kenar çubuğu: yükleme ve ayarlar -------------------------------------------
with st.sidebar:
    st.subheader("Notların")
    yuklenenler = st.file_uploader(
        "PDF, TXT veya MD dosyalarını yükle", type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help=f"En fazla {MAKS_DOSYA} dosya ve toplam {MAKS_SAYFA} sayfa. "
             "Dosyaların yalnızca senin oturumunda işlenir, kaydedilmez.")

    anahtar = api_anahtari()
    if not anahtar:
        anahtar = st.text_input(
            "Groq API anahtarı", type="password",
            help="Ücretsiz anahtar: console.groq.com/keys. Anahtar yalnızca bu "
                 "oturumda kullanılır, kaydedilmez.") or None

    st.divider()
    st.subheader("Ayarlar")
    k = st.slider("Getirilecek parça sayısı", min_value=2, max_value=10, value=rag.K,
                  help="Her soruda notlardan kaç parça bulunup modele verilsin. "
                       "Cevaplar eksik kalıyorsa artır.")
    if st.button("Sohbeti temizle", icon=":material/delete_sweep:", use_container_width=True):
        st.session_state.mesajlar = []
    st.caption(f"Model: {rag.GROQ_MODEL}")

# --- Üst bilgi -----------------------------------------------------------------
def ust_bilgi(etiketler: list[str]):
    etiket_html = "".join(f'<span class="da-etiket">{e}</span>' for e in etiketler)
    st.markdown(
        '<p class="da-baslik">Ders Asistanı</p>'
        '<p class="da-alt">Notlarını yükle, sorularını sor. Cevaplar yalnızca '
        'yüklediğin notlara dayanır ve kaynak sayfasıyla gelir.</p>'
        f'<div class="da-etiketler">{etiket_html}</div>',
        unsafe_allow_html=True)


# --- Boş durum: henüz dosya yok --------------------------------------------------
if not yuklenenler:
    ust_bilgi([])
    with st.container(border=True):
        st.markdown("**Başlamak için notlarını yükle.**")
        st.markdown(
            "1. Soldaki **Notların** bölümüne PDF, TXT veya MD dosyalarını sürükle.\n"
            "2. Sistem notları parçalara bölüp aramaya hazırlar (birkaç saniye ile birkaç dakika).\n"
            "3. Alttaki kutuya sorunu yaz; cevap, hangi dosyanın hangi sayfasına "
            "dayandığıyla birlikte gelir.")
        st.caption("Taranmış (fotoğraf) PDF'ler okunamaz; metni seçilebilen PDF'ler gerekir.")
    st.session_state.mesajlar = []
    st.session_state.pop("imza", None)
    st.stop()

if len(yuklenenler) > MAKS_DOSYA:
    ust_bilgi([])
    st.error(f"En fazla {MAKS_DOSYA} dosya yükleyebilirsin; şu an {len(yuklenenler)} dosya var.",
             icon=":material/error:")
    st.stop()

# --- Notları işle (aynı dosyalar için önbellekten gelir) -------------------------
dosyalar = tuple(sorted((d.name, d.getvalue()) for d in yuklenenler))
imza = tuple((ad, len(veri)) for ad, veri in dosyalar)
if st.session_state.get("imza") != imza:  # Dosyalar değiştiyse eski sohbet geçersiz
    st.session_state.imza = imza
    st.session_state.mesajlar = []

with st.spinner(f"{len(dosyalar)} dosya okunuyor ve aramaya hazırlanıyor..."):
    indeks = indeks_kur(dosyalar)

if "hata" in indeks:
    ust_bilgi([])
    st.error(indeks["hata"], icon=":material/error:")
    st.stop()

X, parcalar = indeks["X"], indeks["parcalar"]
ust_bilgi([f"{len(indeks['dosya'])} döküman", f"{indeks['sayfa']} sayfa",
           f"{len(parcalar)} parça"])
if indeks["bos"] > indeks["sayfa"] / 2:
    st.warning("Sayfaların çoğundan metin çıkmadı. Bazı PDF'ler taranmış (resim) "
               "olabilir; bu sayfalardaki bilgiye ulaşılamaz.", icon=":material/warning:")

if not anahtar:
    st.info("Soru sorabilmek için soldaki kutuya bir Groq API anahtarı gir.",
            icon=":material/key:")
    st.stop()
llm = groq_istemcisi(anahtar)

# --- Sohbet ----------------------------------------------------------------------
soru = st.chat_input("Notlarına bir soru sor...")
if soru:
    st.session_state.mesajlar.append({"rol": "user", "metin": soru})

if not st.session_state.mesajlar:
    st.caption("Notların hazır. İlk sorunu alttaki kutuya yaz.")

for m in st.session_state.mesajlar:
    with st.chat_message(m["rol"], avatar=AVATAR[m["rol"]]):
        st.markdown(m["metin"])
        if m.get("kaynaklar"):
            kaynaklari_goster(m["kaynaklar"])

if soru:
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        with st.spinner("Notlarda aranıyor ve cevap yazılıyor..."):
            bulunanlar = rag.ara(soru, embedding_modeli(), X, parcalar, k=k)
            try:
                cevap = latex_duzelt(rag.cevapla(soru, bulunanlar, llm))
            except Exception as hata:
                cevap = (f"Modelden cevap alınamadı: `{hata}`\n\n"
                         "Anahtar geçersiz olabilir ya da model adı eskimiş olabilir "
                         "(rag.py içindeki GROQ_MODEL).")
        st.markdown(cevap)
        kaynaklari_goster(bulunanlar)

    st.session_state.mesajlar.append(
        {"rol": "assistant", "metin": cevap, "kaynaklar": bulunanlar})
