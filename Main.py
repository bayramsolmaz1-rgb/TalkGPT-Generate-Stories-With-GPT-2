import streamlit as st
from transformers import pipeline

# Sayfa yapılandırması
st.set_page_config(page_title="TalkGPT", page_icon="🤖")
st.title("TalkGPT | Generate Stories With GPT-2")

# --- Yan Menü: Model Seçimi ---
model_choice = st.sidebar.selectbox("Model Seç", [
    "ytu-ce-cosmos/turkish-gpt2", # Türkçe için en iyisi
    "dbmdz/gpt2-turkish-cased",   # Alternatif Türkçe
    "gpt2",                       # Orijinal İngilizce
    "distilgpt2"                  # Hafif ve hızlı
])

# Modeli her seferinde yeniden yüklememesi için önbelleğe alıyoruz
@st.cache_resource
def load_model(m_name):
    with st.spinner(f"{m_name} yükleniyor, lütfen bekleyin..."):
        return pipeline('text-generation', model=m_name)

# Seçilen modeli yükle
generator = load_model(model_choice)

# --- Yanıt Üretme Fonksiyonu ---
def get_gpt2_response(prompt):
    # Parametreler: top_q -> top_p olarak düzeltildi
    res = generator(
        prompt, 
        max_length=200,          # Hikaye için uzunluğu artırdık
        do_sample=True, 
        top_p=0.92,              # Doğru parametre ismi
        top_k=50,
        no_repeat_ngram_size=3,  # Tekrarı önlemek için kritik
        repetition_penalty=1.5   # Kelime yozlaşmasını engeller
    )
    return res[0]['generated_text']

# --- Sohbet Geçmişi ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Kullanıcı Etkileşimi ---
if prompt := st.chat_input("Bir hikaye başlatın veya soru sorun..."):
    # 1. Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Modelden yanıt üret
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            try:
                full_output = get_gpt2_response(prompt)
                st.markdown(full_output)
                # 3. Yanıtı geçmişe kaydet
                st.session_state.messages.append({"role": "assistant", "content": full_output})
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
