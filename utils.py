import io
import pickle
from PIL import Image
import numpy as np
import onnxruntime as ort


def load_all_resources():
  session = ort.InferenceSession("mobilenet.onnx")
  with open("jewellery_metadata (1).pkl", "rb") as f:
    data = pickle.load(f)

  catalog_features = np.array(data["features"], dtype=np.float32)

  # 1. تطبيع متجهات الكتالوج ليكون طول كل متجه = 1
  norms = np.linalg.norm(catalog_features, axis=1, keepdims=True)
  norms[norms == 0] = 1e-10
  catalog_features = catalog_features / norms

  return session, catalog_features, data["images"]


def extract_embedding(img_pil, session):
  img = img_pil.resize((224, 224))
  img_arr = np.array(img, dtype=np.float32)
  img_arr = (img_arr / 127.5) - 1.0
  img_arr = np.expand_dims(img_arr, axis=0)

  input_name = session.get_inputs()[0].name
  embedding = session.run(None, {input_name: img_arr})[0].flatten()

  # 2. تطبيع متجه الصورة المدخلة ليكون طوله = 1
  norm = np.linalg.norm(embedding)
  return embedding / (norm if norm > 0 else 1e-10)


def search_similar_items(
    query_feature, catalog_features, catalog_images, threshold=0.30, top_k=20
):
  # 3. حساب الـ Cosine Similarity بعد التطبيع (النتيجة ستكون بين 0.0 و 1.0)
  similarities = np.dot(catalog_features, query_feature)

  # ضمان عدم تجاوز النطاق [0, 1] بسبب الكسور العشرية
  similarities = np.clip(similarities, 0.0, 1.0)

  top_indices = np.argsort(similarities)[::-1]

  results = []
  for idx in top_indices:
    score = float(similarities[idx])
    if score >= threshold:
      raw_img = Image.open(io.BytesIO(catalog_images[idx])).convert("RGB")
      results.append((score, raw_img))
    if len(results) == top_k:
      break

  return results