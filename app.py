from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
DATA_PATH = Path(__file__).parent / "movies.csv"

def load_movies():
    df = pd.read_csv(DATA_PATH).fillna("")
    # Build a text profile from genres, description, language, and actors.
    df["features"] = (
        df["title"].astype(str) + " " +
        df["language"].astype(str) + " " +
        df["genres"].astype(str) + " " +
        df["actors"].astype(str) + " " +
        df["description"].astype(str)
    )
    return df

movies = load_movies()
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["features"])
similarity_matrix = cosine_similarity(tfidf_matrix)

def movie_to_dict(row):
    return {
        "title": row["title"],
        "year": row["year"],
        "language": row["language"],
        "genres": row["genres"],
        "actors": row["actors"],
        "platforms": row["platforms"],
        "description": row["description"],
    }

def recommend_similar(title, limit=6):
    matches = movies.index[movies["title"].str.casefold() == title.casefold()].tolist()
    if not matches:
        return []
    idx = matches[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    results = []
    for movie_idx, score in scores:
        if movie_idx == idx:
            continue
        row = movies.iloc[movie_idx]
        item = movie_to_dict(row)
        item["score"] = round(float(score) * 100)
        results.append(item)
        if len(results) >= limit:
            break
    return results

@app.route("/", methods=["GET", "POST"])
def index():
    query = request.values.get("q", "").strip()
    language = request.values.get("language", "All")
    genre = request.values.get("genre", "All")
    mode = request.values.get("mode", "related")
    results = []
    selected_movie = None
    message = ""

    if request.method == "POST" or query or language != "All" or genre != "All":
        filtered = movies.copy()

        if language != "All":
            filtered = filtered[filtered["language"].str.casefold() == language.casefold()]
        if genre != "All":
            filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]

        if query:
            # First identify whether the query is an actor. If so, show all matching catalogue titles.
            actor_mask = filtered["actors"].str.contains(query, case=False, na=False, regex=False)
            actor_results = filtered[actor_mask]
            if mode == "actor" or not actor_results.empty:
                if not actor_results.empty:
                    results = [movie_to_dict(row) for _, row in actor_results.iterrows()]
                    message = f"Movies featuring actor matching “{query}”"
                elif mode == "actor":
                    message = f"No actor matches found for “{query}”. Try a shorter name."
            if not results:
                title_matches = filtered[filtered["title"].str.contains(query, case=False, na=False, regex=False)]
                if not title_matches.empty:
                    selected_row = title_matches.iloc[0]
                    selected_movie = movie_to_dict(selected_row)
                    if mode == "related":
                        results = recommend_similar(selected_row["title"])
                        message = f"Movies similar to “{selected_row['title']}” (TF-IDF + cosine similarity)"
                    else:
                        results = [movie_to_dict(row) for _, row in title_matches.iterrows()]
                        message = f"Movie search results for “{query}”"
                else:
                    # Search genre, language, description, or title/actor text.
                    searchable = filtered[
                        filtered["genres"].str.contains(query, case=False, na=False, regex=False) |
                        filtered["language"].str.contains(query, case=False, na=False, regex=False) |
                        filtered["description"].str.contains(query, case=False, na=False, regex=False) |
                        filtered["title"].str.contains(query, case=False, na=False, regex=False)
                    ]
                    results = [movie_to_dict(row) for _, row in searchable.iterrows()]
                    message = f"Matching movies for “{query}”"
        else:
            results = [movie_to_dict(row) for _, row in filtered.iterrows()]
            message = "Movies matching your filters"
    else:
        results = [movie_to_dict(row) for _, row in movies.head(8).iterrows()]
        message = "Popular sample movies — search a title or actor to begin"

    return render_template(
        "index.html",
        results=results,
        selected_movie=selected_movie,
        query=query,
        language=language,
        genre=genre,
        mode=mode,
        message=message,
        movie_titles=movies["title"].tolist(),
        languages=sorted(movies["language"].unique().tolist()),
        genres=["Comedy", "Romantic", "Emotional", "Horror", "Action", "Drama", "Thriller", "Family", "Adventure", "Sci-Fi", "Mystery"],
    )

if __name__ == "__main__":
    app.run(debug=True)
