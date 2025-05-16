import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from scipy.stats import gaussian_kde
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📍 Interactive Heatmap Overlay")

# -- CONFIG --
CSV_PATH = "/content/gdrive/Shareddrives/emazing/Studies/1-Active - 2025 projects/PowerBI/data/df_heatmaps.csv"
IMAGE_FOLDER = "heatmap_images"
QUESTION_ID = 7633
EXP_NAME = "Control"
Q_LABEL = "LIKES"
USER_GROUP = "all users"
# ------------


def find_image_file(question_id, image_dir=IMAGE_FOLDER):
    for ext in ['.png', '.jpg']:
        file_path = os.path.join(image_dir, f"{question_id}{ext}")
        if os.path.isfile(file_path):
            return file_path
    return None

# Load the CSV directly
df = pd.read_csv(CSV_PATH)
qid = int(QUESTION_ID)

group = df.query("question_id == @qid")
img_path = find_image_file(qid)

if not img_path:
    st.warning(f"Image for question_id {qid} not found in '{IMAGE_FOLDER}' folder.")
else:
    with Image.open(img_path) as img:
        width, height = img.size
        coords = []
        texts = []

        for _, row in group.iterrows():
            try:
                answer_data = json.loads(row["answer"])
                x = float(answer_data["x"]) / 100 * width
                y = float(answer_data["y"]) / 100 * height
                coords.append([x, y])
                texts.append(answer_data.get("text", ""))
            except Exception as e:
                continue

        coords = np.array(coords)
        if len(coords) == 0:
            st.warning("No valid coordinates found.")
        else:
            kde = gaussian_kde(coords.T)
            densities = kde(coords.T)
            norm_densities = (densities - densities.min()) / (densities.max() - densities.min())

            plot_df = pd.DataFrame(coords, columns=['x', 'y'])
            plot_df['density'] = norm_densities
            plot_df['text'] = texts

            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color='density',
                hover_name='text',
                color_continuous_scale='Jet',
                title=f"Density-Colored Scatter for {EXP_NAME} {Q_LABEL} ({USER_GROUP})",
                width=1000,
                height=600
            )

            fig.update_layout(
                images=[dict(
                    source=img,
                    x=0,
                    y=0,
                    sizex=width,
                    sizey=height,
                    xref="x",
                    yref="y",
                    sizing="stretch",
                    opacity=0.5,
                    layer="below"
                )],
                yaxis=dict(scaleanchor="x", autorange='reversed'),
                xaxis=dict(constrain='domain')
            )

            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='black')))
            st.plotly_chart(fig, use_container_width=True)
