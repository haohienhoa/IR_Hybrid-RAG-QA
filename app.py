import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/chat" 

st.set_page_config(page_title="Trợ lý Pháp luật AI", page_icon="⚖️", layout="centered")
st.title("Trợ lý Pháp luật Việt Nam")
st.caption("Giao diện Frontend - Kết nối với FastAPI Backend")

# --- LỊCH SỬ CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- GIAO DIỆN NHẬP & GỌI API ---
if prompt := st.chat_input("Nhập tình huống pháp lý..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang gửi yêu cầu đến Server AI..."):
            try:
                # GỌI API BACKEND FASTAPI BẰNG THƯ VIỆN REQUESTS
                response = requests.post(
                    API_URL, 
                    json={"question": prompt},
                    timeout=60 # Đợi tối đa 60 giây
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "Không nhận được phản hồi.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Lỗi Server: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Không thể kết nối đến Backend API. Vui lòng kiểm tra xem FastAPI đã được bật chưa!")