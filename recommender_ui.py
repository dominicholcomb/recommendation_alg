import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler

# === Load & Preprocess Data ===
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("steam-200k.csv")
    df.columns = ['user_id', 'game_name', 'purchase_type', 'hours_played', 'na']

    # Aggregate play data per user/game
    df = df.groupby(['user_id', 'game_name']).agg({
        'purchase_type': 'max',
        'hours_played': 'sum'
    }).reset_index()

    # Robust scaling to reduce outlier impact
    scaler = RobustScaler(with_centering=False, with_scaling=True)
    df['hours_played'] = scaler.fit_transform(df[['hours_played']])

    # Quantile thresholds for recommendation impact bounds
    lower_bound = df["hours_played"].quantile(0.75)
    upper_bound = df["hours_played"].quantile(0.95)

    return df, lower_bound, upper_bound

df, lower_bound, upper_bound = load_and_preprocess_data()

# === UI Setup ===
st.title("🎮 Steam Game Recommender")

game_list = sorted(df['game_name'].unique())
selected_games = st.multiselect(
    "Type in 1–5 games you like. The first games will be weighted more heavily:",
    options=game_list,
    max_selections=5,
    key="game_selector"
)

# === Generate Recommendations on Button Click ===
if st.button("Get Recommendations") and selected_games:
    dummy_user_id = 999999999

    # Weighted input hours: earlier games have higher impact
    default_hours = [20, 10, 10, 8, 3]
    input_hours = default_hours[:len(selected_games)]

    dummy_entries = pd.DataFrame({
        'user_id': [dummy_user_id] * len(selected_games),
        'game_name': selected_games,
        'purchase_type': ['purchase'] * len(selected_games),
        'hours_played': input_hours
    })

    # Inject dummy user into a clean copy of the dataset
    df_cleaned = df[df['user_id'] != dummy_user_id]
    df_extended = pd.concat([df_cleaned, dummy_entries], ignore_index=True)

    # Create matrices
    user_item_matrix = df_extended.pivot_table(index='user_id', columns='game_name', values='hours_played', fill_value=0)
    user_similarity_matrix = cosine_similarity(user_item_matrix)
    user_similarity_matrix = pd.DataFrame(user_similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # === Recommendation Logic ===
    def recommend_games(user_id, user_similarity_matrix, user_item_matrix, top_n=10, closest_users=10):
        user_similarities = user_similarity_matrix.loc[user_id].drop(user_id)
        closest_users_scores = user_similarities.nlargest(closest_users)
        played_games = user_item_matrix.loc[user_id]
        played_games = played_games[played_games > 0].index

        recommendations = {}
        for other_user, similarity_score in closest_users_scores.items():
            other_user_games = user_item_matrix.loc[other_user]
            for game, hours_played in other_user_games.items():
                if game not in played_games and hours_played > 0:
                    if game not in recommendations:
                        recommendations[game] = 0
                    # Bound hours_played to avoid extreme outliers
                    bounded_playtime = max(min(hours_played, upper_bound), lower_bound)
                    recommendations[game] += bounded_playtime * similarity_score

        recommended_games = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [game for game, _ in recommended_games][:top_n]

    # === Show Output ===
    recommendations = recommend_games(dummy_user_id, user_similarity_matrix, user_item_matrix)

    st.subheader("✨ Recommended Games:")
    for game in recommendations:
        st.markdown(f"- **{game}**")

elif not selected_games:
    st.info("Pick some games you like to get started!")
