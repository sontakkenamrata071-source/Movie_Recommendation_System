MOVIE FINDER - PYTHON MACHINE LEARNING MINI PROJECT

Requirements:
- Python 3.10 or newer recommended
- Internet connection for first-time package installation

RUN ON WINDOWS:
1. Extract this ZIP file to a folder.
2. Open that folder in File Explorer.
3. Click the address bar, type cmd, and press Enter.
4. Run:
   py -m pip install -r requirements.txt
5. Run:
   py app.py
6. Open the local URL shown in the terminal, usually:
   http://127.0.0.1:5000

If the 'py' command is unavailable, try 'python' instead:
   python -m pip install -r requirements.txt
   python app.py

ML:
- TF-IDF Vectorizer converts combined movie text to numerical features.
- Cosine similarity compares movie feature vectors.
- Actor search is a catalogue lookup, while movie recommendations use the similarity model.

The included movies.csv is a small sample dataset. Streaming services are suggestions to check, not live-confirmed availability.
