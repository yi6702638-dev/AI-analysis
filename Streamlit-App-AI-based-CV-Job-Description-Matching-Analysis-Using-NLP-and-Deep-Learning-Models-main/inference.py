import json
import re
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cleanResume(resumeText: str) -> str:
    """
    Clean resume/JD text similarly to the notebook (URLs, mentions, hashtags, punctuation, non-ascii, extra spaces).
    """
    resumeText = re.sub(r"http\S+\s*", " ", resumeText)
    resumeText = re.sub(r"RT|cc", " ", resumeText)
    resumeText = re.sub(r"#\S+", "", resumeText)
    resumeText = re.sub(r"@\S+", "  ", resumeText)
    resumeText = re.sub(r"[%s]" % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", resumeText)
    resumeText = re.sub(r"[^\x00-\x7f]", r" ", resumeText)
    resumeText = re.sub(r"\s+", " ", resumeText).strip()
    return resumeText


# A lightweight skill list (can expand/replace with your notebook list if have one).
SKILLS_LIST = [
    "Python","Java","C++","C#","R","PHP","JavaScript","TypeScript","HTML","CSS",
    "Django","Flask","Spring","React","Angular","Node","ASP.NET","TensorFlow","Keras","PyTorch",
    "Hadoop","Spark","Hive","Docker","Kubernetes","AWS","Azure","GCP",
    "SQL","MySQL","PostgreSQL","MongoDB","Oracle","Git","Tableau","Power BI","Excel",
    "MATLAB","SolidWorks","AutoCAD",
    "Selenium","Jenkins","CI/CD","Agile","Scrum",
    "Project Management","Team Management","Leadership","Communication",
    "Salesforce","Negotiation","Legal Research","Litigation","Nutrition",
]
_SKILLS_LOWER = [s.lower() for s in SKILLS_LIST]


def extract_skills(text: str):
    text_l = text.lower()
    found = []
    for s in _SKILLS_LOWER:
        if re.search(r"\b" + re.escape(s) + r"\b", text_l):
            found.append(s)
    return sorted({x.title() for x in found})


import json
import numpy as np
import keras
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class InferenceEngine:
    def __init__(self, artifacts_dir="artifacts"):
        # ---- Load vectorizer SavedModel as inference-only layer ----
        self.vec_layer = keras.layers.TFSMLayer(
            f"{artifacts_dir}/vectorizer",
            call_endpoint="serving_default"
        )

        # ---- Load models as inference-only layers ----
        self.mc_layer = keras.layers.TFSMLayer(
            f"{artifacts_dir}/model_mc",
            call_endpoint="serving_default"
        )
        self.bin_layer = keras.layers.TFSMLayer(
            f"{artifacts_dir}/model_bin",
            call_endpoint="serving_default"
        )
        self.match_layer = keras.layers.TFSMLayer(
            f"{artifacts_dir}/model_match",
            call_endpoint="serving_default"
        )
        
        self.match_fn = tf.saved_model.load(f"{artifacts_dir}/model_match").signatures["serving_default"]

        with open(f"{artifacts_dir}/label_classes.json", "r", encoding="utf-8") as f:
            self.label_classes = json.load(f)

    def _vectorize(self, text_list):
        out = self.vec_layer(np.array(text_list, dtype=object))
        if isinstance(out, dict):
            out = list(out.values())[0]
        out = tf.cast(out, tf.int32)
        return out

    def _call_layer(self, layer, inputs):
        if isinstance(inputs, dict):
            out = layer(**inputs)   
        else:
            out = layer(inputs)
    
        if isinstance(out, dict):
            out = list(out.values())[0]
        return out



    def predict_category(self, cv_text: str):
        cv = cleanResume(cv_text)
        X = self._vectorize([cv])
        probs = self._call_layer(self.mc_layer, X).numpy()[0]
        top_idx = int(np.argmax(probs))
        return {
            "top_category": self.label_classes[top_idx],
            "top_prob": float(probs[top_idx]),
            "probs": {self.label_classes[i]: float(probs[i]) for i in np.argsort(probs)[::-1][:10]}
        }

    def predict_shortlist(self, cv_text: str):
        cv = cleanResume(cv_text)
        X = self._vectorize([cv])
        p = float(self._call_layer(self.bin_layer, X).numpy()[0][0])
        return {"shortlist_prob": p, "decision": "Shortlist" if p >= 0.5 else "Reject"}

    def predict_match(self, cv_text: str, jd_text: str):
        cv = cleanResume(cv_text)
        jd = cleanResume(jd_text)
    
        X_cv = self._vectorize([cv])  # (1, 300) int32
        X_jd = self._vectorize([jd])  # (1, 300) int32
    

        out = self.match_fn(jd_input=X_jd, resume_input=X_cv)
    

        y = list(out.values())[0]
        p = float(y.numpy()[0][0])
        return {"match_prob": p}


    def run_all(self, cv_text: str, jd_text: str):
        # skills
        skills_cv = extract_skills(cv_text) if "extract_skills" in globals() else []
        skills_jd = extract_skills(jd_text) if "extract_skills" in globals() else []
        overlap = sorted(set(skills_cv) & set(skills_jd))
    
        # model outputs
        out = {
            "category": self.predict_category(cv_text),
            "shortlist": self.predict_shortlist(cv_text),
            "match": self.predict_match(cv_text, jd_text),
            "skills": {
                "cv_skills": skills_cv,
                "jd_skills": skills_jd,
                "overlap": overlap
            }
        }
    
        # TF-IDF similarity (optional but nice)
        try:
            cv = cleanResume(cv_text)
            jd = cleanResume(jd_text)
            vect = TfidfVectorizer(stop_words="english", max_features=5000)
            M = vect.fit_transform([cv, jd])
            sim = float(cosine_similarity(M[0], M[1])[0][0])
            out["similarity"] = {"tfidf_cosine": sim}
        except Exception as e:
            out["similarity"] = {"tfidf_cosine": None, "error": str(e)}
    
        return out

