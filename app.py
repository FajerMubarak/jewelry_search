from PIL import Image
import streamlit as st
from utils import extract_embedding, load_all_resources, search_similar_items

# 1. إعداد الصفحة
st.set_page_config(page_title="Khazeena AI | Visual Search", layout="wide")


# 2. تحميل البيانات لمرة واحدة فقط
@st.cache_resource(show_spinner="Loading Khazeena AI Catalog...")
def get_cached_resources():
  return load_all_resources()


session, catalog_features, catalog_images = get_cached_resources()

# 3. العنوان الرئيسي
st.title("💎 Khazeena Visual Search")
st.write("Upload or capture a jewelry piece to discover matching catalog items.")

# 4. التبويبات لإدخال الصورة (Upload / Camera)
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Take Photo"])

query_img = None

with tab_upload:
  uploaded_file = st.file_uploader(
      "Choose a jewelry image...", type=["jpg", "jpeg", "png"]
  )
  if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")

with tab_camera:
  camera_file = st.camera_input("Take a photo")
  if camera_file:
    query_img = Image.open(camera_file).convert("RGB")

# عرض الصورة المدخلة
if query_img:
  st.markdown("---")
  st.image(query_img, caption="Selected Query Image", width=220)

  # 5. البحث وعرض النتائج
  st.subheader("Top Matching Products")

  query_feature = extract_embedding(query_img, session)
  results = search_similar_items(
      query_feature, catalog_features, catalog_images
  )

  if not results:
    st.warning("No matching items found for the uploaded image.")
  else:
    cols = st.columns(5)
    for idx, (score, img_obj) in enumerate(results):
      with cols[idx % 5]:
        st.image(img_obj, use_container_width=True)
        st.caption(f"Similarity Score: {score:.1%}")