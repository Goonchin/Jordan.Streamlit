import streamlit as st
from fastai.vision.all import *
import gdown


st.markdown("""# Sneaker Shoe Type Classifier 
Are you a sneaker enthusiast? Ever been out somewhere and you spot someone with some cool sneakers? But have no idea the brand or even shoe type? Take a picture, upload it here and chances are you'll end up knowing exactly what they are. Try it out!""")

st.markdown("""### Upload your shoe image here""")

image_file = st.file_uploader("Image Uploader", type=["png","jpg","jpeg"])

## Model Loading Section
model_path = Path("export.pkl")

if not model_path.exists():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        url = 'https://drive.google.com/file/d/1-8LtQQ33ljd7ljmmdN3_nVi_HFG2LnsQ/view?usp=share_link'
        output = 'export.pkl'
        gdown.download(url, output, quiet=False)
    learn_inf = load_learner('export.pkl')
else:
    learn_inf = load_learner('export.pkl')
 
col1, col2 = st.columns(2)
if image_file is not None:
    img = PILImage.create(image_file)
    pred, pred_idx, probs = learn_inf.predict(img)

    with col1:
        st.markdown(f"""### Predicted shoe: {pred.capitalize()}""")
        st.markdown(f"""### Accuracy: {round(max(probs.tolist()), 3) * 100}%""")
    with col2:
        st.image(img, width=300)
