"""
Run this INSIDE your notebook environment AFTER training finishes to export artifacts.

Usage (in notebook cell):
  !python export_artifacts_from_notebook.py

Prereqs:
  notebook should define:
    - model_mc, model_bin, model_match : trained tf.keras Models
    - vectorizer : a tf.keras.layers.TextVectorization already adapted
    - label_enc : sklearn LabelEncoder fitted on your category labels
"""
import os, json
import tensorflow as tf

def main():

    global model_mc, model_bin, model_match, vectorizer, label_enc

    os.makedirs("artifacts", exist_ok=True)

    model_mc.save("artifacts/model_mc.keras")
    model_bin.save("artifacts/model_bin.keras")
    model_match.save("artifacts/model_match.keras")

    vec_model = tf.keras.Sequential([vectorizer])
    vec_model.save("artifacts/vectorizer.keras")

    with open("artifacts/label_classes.json", "w", encoding="utf-8") as f:
        json.dump(label_enc.classes_.tolist(), f, ensure_ascii=False, indent=2)

    print("✅ Exported artifacts to ./artifacts/")
    print("Next: copy the artifacts/ folder into resume_app/artifacts/ and run:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
