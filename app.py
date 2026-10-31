import os
import numpy as np
import cv2
import joblib
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import hog, local_binary_pattern

# ZeroGPU (Hugging Face) exige au moins une fonction décorée @spaces.GPU.
# Notre application n'utilise pas de GPU : tout le calcul se fait sur CPU, cette fonction n'est jamais appelée.
try:
    import spaces

    @spaces.GPU
    def _fonction_gpu_non_utilisee():
        return None
except ImportError:
    pass  # en dehors de Hugging Face (Colab, local), le module "spaces" n'existe pas : on ignore
    
# ---------- Chargement des modèles et des paramètres ----------
DOSSIER = os.path.dirname(os.path.abspath(__file__))
donnees = joblib.load(os.path.join(DOSSIER, "modeles_visages.joblib"))
MODELE_SEXE, MODELE_AGE = donnees["sexe"], donnees["age"]
CALIB = donnees["calibration"]
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
POLICE = os.path.join(DOSSIER, "DejaVuSans.ttf")

NOMS = {("Male", "Enfant"): "Garçon", ("Female", "Enfant"): "Fille",
        ("Male", "Adulte"): "Homme", ("Female", "Adulte"): "Femme",
        ("Male", "Âgé"): "Vieil homme", ("Female", "Âgé"): "Vieille femme"}

# ---------- 1. Recadrage calibré (même cadrage qu'UTKFace) ----------
def recadrer(img, boite):
    x, y, bw, bh = boite
    cote = CALIB["rapport"] * bw
    cx, cy = x + bw / 2 + CALIB["dx"] * bw, y + bh / 2 + CALIB["dy"] * bh
    x0, y0 = int(round(cx - cote / 2)), int(round(cy - cote / 2))
    x1, y1 = int(round(cx + cote / 2)), int(round(cy + cote / 2))
    p = max(0, -x0, -y0, x1 - img.shape[1], y1 - img.shape[0])
    if p > 0:
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT)
    return cv2.resize(img[y0 + p:y1 + p, x0 + p:x1 + p], (100, 100))

# ---------- 2. Segmentation du visage (peau + GrabCut contraint) ----------
def segmenter(img, iterations=3, seuil_min=0.30):
    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    peau = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb), (0, 133, 77), (255, 173, 127)) > 0
    grande = np.zeros((h, w), np.uint8)
    cv2.ellipse(grande, centre, (int(w * 0.45), int(h * 0.55)), 0, 0, 360, 1, -1)
    petite = np.zeros((h, w), np.uint8)
    cv2.ellipse(petite, centre, (int(w * 0.20), int(h * 0.25)), 0, 0, 360, 1, -1)
    masque = np.full((h, w), cv2.GC_BGD, np.uint8)
    masque[grande == 1] = cv2.GC_PR_BGD
    masque[(grande == 1) & peau] = cv2.GC_PR_FGD
    masque[petite == 1] = cv2.GC_FGD
    cv2.grabCut(img, masque, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
    res = np.where((masque == cv2.GC_FGD) | (masque == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros((h, w), np.uint8)
    if contours:
        cv2.drawContours(final, [max(contours, key=cv2.contourArea)], -1, 255, thickness=-1)
    if final.mean() / 255 < seuil_min:
        final = grande * 255
    return final

# ---------- 3. Caractéristiques (HOG + LBP + couleur) ----------
def caracteristiques(img, masque):
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    peau = masque > 0
    f_hog = hog(cv2.resize(gris, (64, 64)), orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(gris, P=8, R=1, method="uniform")
    f_lbp = []
    for by in range(4):
        for bx in range(4):
            zone = (slice(by * 25, (by + 1) * 25), slice(bx * 25, (bx + 1) * 25))
            hist, _ = np.histogram(lbp[zone][peau[zone]], bins=10, range=(0, 10))
            f_lbp.append(hist / hist.sum() if hist.sum() > 0 else hist.astype(float))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    f_col = []
    for canal, classes, maximum in [(0, 16, 180), (1, 8, 256), (2, 8, 256)]:
        hist = cv2.calcHist([hsv], [canal], masque, [classes], [0, maximum]).flatten()
        f_col.append(hist / hist.sum() if hist.sum() > 0 else hist)
    return np.hstack([f_hog, np.concatenate(f_lbp), np.concatenate(f_col)]).reshape(1, -1)

# ---------- 4. Analyse complète d'une photo ----------
def analyser(image):
    if image is None:
        return None, [], "Veuillez déposer une photo ou utiliser la webcam."
    img = np.ascontiguousarray(image[..., :3]).astype(np.uint8)
    # Les grandes photos sont réduites pour accélérer l'analyse
    echelle = 1000 / max(img.shape[:2])
    if echelle < 1:
        img = cv2.resize(img, None, fx=echelle, fy=echelle)

    gris = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    boites = CASCADE.detectMultiScale(gris, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    aucun_visage = len(boites) == 0
    if aucun_visage:
        boites = [(0, 0, img.shape[1], img.shape[0])]

    annotee = Image.fromarray(img.copy())
    dessin = ImageDraw.Draw(annotee)
    police = ImageFont.truetype(POLICE, size=max(14, img.shape[1] // 40))
    galerie = []
    lignes = ["| Visage | Catégorie | Confiance sexe | Confiance âge |", "|---|---|---|---|"]

    for k, (x, y, bw, bh) in enumerate(sorted(boites, key=lambda b: b[0]), start=1):
        visage = cv2.resize(img, (100, 100)) if aucun_visage else recadrer(img, (x, y, bw, bh))
        masque = segmenter(visage)
        f = caracteristiques(visage, masque)
        p_sexe, p_age = MODELE_SEXE.predict_proba(f)[0], MODELE_AGE.predict_proba(f)[0]
        categorie = NOMS[(MODELE_SEXE.classes_[p_sexe.argmax()], MODELE_AGE.classes_[p_age.argmax()])]

        texte = f"{k}. {categorie}"
        dessin.rectangle([x, y, x + bw, y + bh], outline=(0, 200, 0), width=3)
        g, h_, d, b = dessin.textbbox((x, y + bh + 4), texte, font=police)
        dessin.rectangle([g - 3, h_ - 2, d + 3, b + 2], fill=(0, 120, 0))
        dessin.text((x, y + bh + 4), texte, fill=(255, 255, 255), font=police)

        galerie.append((cv2.resize(visage, (200, 200)), f"Visage {k} : recadré"))
        galerie.append((cv2.resize(cv2.bitwise_and(visage, visage, mask=masque), (200, 200)), f"Visage {k} : segmenté"))
        lignes.append(f"| {k} | **{categorie}** | {100 * p_sexe.max():.0f} % | {100 * p_age.max():.0f} % |")

    resume = "\n".join(lignes)
    if aucun_visage:
        resume = "⚠️ Aucun visage détecté : la photo entière a été analysée comme un visage.\n\n" + resume
    return np.array(annotee), galerie, resume

# ---------- 5. Interface Gradio ----------
demo = gr.Interface(
    fn=analyser,
    inputs=gr.Image(type="numpy", sources=["upload", "webcam"], label="Photo"),
    outputs=[gr.Image(label="Résultat"),
             gr.Gallery(label="Visages recadrés et segmentés", columns=4, height="auto"),
             gr.Markdown(label="Récapitulatif")],
    title="Fille, garçon, femme, homme, vieille femme ou vieil homme ?",
    description=("Classification du **sexe** et de la **tranche d'âge** par méthodes **classiques** de vision par ordinateur : "
                 "détection de Haar (Viola-Jones) → segmentation GrabCut → descripteurs HOG, LBP et couleur → SVM. "
                 "Projet du cours de Computer Vision — SUP'RH."),
    article=("**Limites :** le modèle prédit une *apparence perçue* à partir d'annotations binaires du dataset UTKFace. "
             "Il ne détermine ni l'identité de genre ni l'âge réel, et ses performances peuvent varier selon les personnes "
             "(≈ 80 % pour le sexe, ≈ 84 % pour la tranche d'âge). Les photos sont traitées en mémoire et ne sont pas "
             "conservées par l'application."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
