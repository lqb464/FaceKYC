"""Analyst-facing FaceKYC review portal."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("FACEKYC_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_SECONDS = 120


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_verify(id_file: Any, selfie_file: Any) -> dict[str, Any]:
    files = {
        "id_image": (id_file.name, id_file.getvalue(), id_file.type),
        "selfie_image": (selfie_file.name, selfie_file.getvalue(), selfie_file.type),
    }
    response = requests.post(f"{API_URL}/api/v1/verify", files=files, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="FaceKYC Review", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .hero {padding: 1.2rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#146b6f); color:white; margin-bottom:1rem;}
    .scope {padding:.8rem 1rem; border-left:4px solid #d97706; background:#fff7ed; border-radius:6px;}
    [data-testid="stMetric"] {border:1px solid #dbe5e8; padding:.8rem; border-radius:12px; background:white;}
    </style>
    <div class="hero"><h1>FaceKYC Review Console</h1><p>Đối chiếu khuôn mặt 1:1, kiểm tra chất lượng ảnh và sàng lọc presentation attack.</p></div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="scope"><strong>Phạm vi:</strong> Output là tín hiệu hỗ trợ rà soát. '
    "Không phải bằng chứng pháp lý về danh tính; passive PAD từ một ảnh không thay thế "
    "active challenge, NFC/chip verification hoặc quy trình kiểm soát eKYC đầy đủ.</div>",
    unsafe_allow_html=True,
)

try:
    health = api_get("/health")
    model = health.get("model", {})
    api_live = bool(health.get("live"))
    api_ready = bool(health.get("ready"))
except (requests.RequestException, ValueError):
    health, model, api_live, api_ready = {}, {}, False, False

with st.sidebar:
    st.subheader("Trạng thái hệ thống")
    st.success("API đang chạy") if api_live else st.error("Không kết nối được API")
    st.success("Model đã sẵn sàng") if api_ready else st.warning("Model chưa sẵn sàng")
    st.write(f"Model version: `{model.get('model_version', '-')}`")
    st.write(f"Artifact: `{model.get('deployment_status', model.get('artifact_status', '-'))}`")
    if model.get("deployment_status") == "candidate":
        st.warning("Research candidate — không dùng cho production hoặc quyết định KYC tự động.")
    st.caption("Ảnh và embedding không được lưu bởi pipeline mặc định.")

verify_tab, governance_tab, guide_tab = st.tabs(["Xác minh một hồ sơ", "Model governance", "Hướng dẫn chụp"])

with verify_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("1. Ảnh chân dung trên giấy tờ")
        id_file = st.file_uploader("Tải ảnh ID", type=["jpg", "jpeg", "png", "webp"], key="id")
        if id_file:
            st.image(id_file, caption="Ảnh ID", use_container_width=True)
    with right:
        st.subheader("2. Selfie hiện tại")
        selfie_source = st.radio("Nguồn selfie", ["Camera", "Tải ảnh"], horizontal=True)
        if selfie_source == "Camera":
            selfie_file = st.camera_input("Chụp selfie")
        else:
            selfie_file = st.file_uploader("Tải selfie", type=["jpg", "jpeg", "png", "webp"], key="selfie")
        if selfie_file and selfie_source == "Tải ảnh":
            st.image(selfie_file, caption="Selfie", use_container_width=True)

    can_run = api_ready and id_file is not None and selfie_file is not None
    if st.button("Chạy kiểm tra", type="primary", use_container_width=True, disabled=not can_run):
        try:
            with st.spinner("Đang kiểm tra chất lượng, PAD và face match..."):
                st.session_state["facekyc_result"] = api_verify(id_file, selfie_file)
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Không thể xử lý hồ sơ: {detail}")

    if not api_ready:
        st.info("Service đang deployment-blocked cho tới khi bundle PAD đã được đánh giá và phê duyệt.")
    if "facekyc_result" in st.session_state:
        result = st.session_state["facekyc_result"]
        decision = result.get("decision")
        if decision == "verified":
            st.success("Tín hiệu xác minh: ĐẠT — vẫn áp dụng các bước kiểm soát eKYC còn lại.")
        elif decision == "manual_review":
            st.warning("Tín hiệu xác minh: CẦN RÀ SOÁT THỦ CÔNG.")
        elif decision == "recapture":
            st.info("Cần chụp lại ảnh trước khi đánh giá.")
        else:
            st.error("Tín hiệu xác minh: KHÔNG ĐẠT.")
        one, two, three = st.columns(3)
        one.metric("Cosine similarity", f"{result.get('similarity_score', 0):.3f}" if result.get("similarity_score") is not None else "-")
        two.metric("PAD bona-fide score", f"{result.get('liveness_score', 0):.3f}" if result.get("liveness_score") is not None else "-")
        three.metric("Model version", result.get("model_version", "-"))
        st.write("Reason codes:", ", ".join(result.get("reason_codes", [])))
        if result.get("input_warnings"):
            st.warning("Cảnh báo đầu vào: " + ", ".join(result["input_warnings"]))
        with st.expander("Chi tiết chất lượng ảnh"):
            st.json(result.get("image_quality", {}))

with governance_tab:
    st.subheader("Model và dữ liệu đánh giá")
    st.json(model)
    st.markdown(
        """
        - Face verification: FaceNet pretrained VGGFace2; threshold khóa trên LFW fold 8 theo target FMR 1%, đánh giá một lần trên fold 9.
        - Passive PAD proxy: MobileNetV3-Small; threshold khóa theo target APCER 5% trên validation subject-disjoint.
        - PAD chỉ dùng attack tổng hợp từ LFW, không phải benchmark PAD thật và không chứng minh khả năng chống print/replay/mask/deepfake.
        - LFW không đại diện cho ảnh giấy tờ/selfie trong eKYC Việt Nam; bundle hiện chỉ là research candidate.
        - Không ghi log ảnh, embedding hoặc payload sinh trắc học. Monitoring chỉ dùng score tổng hợp, cảnh báo và decision count.
        """
    )

with guide_tab:
    st.subheader("Điều kiện ảnh đầu vào")
    st.markdown(
        """
        - Chỉ một khuôn mặt, nhìn gần thẳng, không bị che khuất.
        - Ánh sáng đều; tránh ngược sáng, ảnh quá tối hoặc cháy sáng.
        - Selfie phải là ảnh mới chụp. Một ảnh tĩnh chỉ hỗ trợ passive PAD, không chứng minh người dùng đang tương tác trực tiếp.
        - Khi hệ thống trả `manual_review` hoặc `recapture`, không tự động chuyển thành từ chối khách hàng.
        """
    )

st.divider()
st.caption("FaceKYC — research-grade biometric decision support. Không dùng độc lập để phê duyệt hoặc từ chối KYC.")
