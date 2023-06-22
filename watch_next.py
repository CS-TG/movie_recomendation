import spacy
from sklearn.metrics.pairwise import cosine_similarity

# Load the language model
nlp = spacy.load('en_core_web_md')

def get_most_similar_movie(description):
    with open('movies.txt', 'r') as file:
        movies = file.readlines()

    # Preprocess the description
    description = description.strip()

    # Calculate similarity scores for each movie
    similarity_scores = []
    for movie in movies:
        movie = movie.strip()
        similarity = cosine_similarity(
            nlp(description).vector.reshape(1, -1),
            nlp(movie).vector.reshape(1, -1)
        )[0][0]
        similarity_scores.append(similarity)

    # Get the index of the most similar movie
    most_similar_index = similarity_scores.index(max(similarity_scores))

    # Return the title of the most similar movie
    return movies[most_similar_index]

# Example usage
description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
most_similar_movie = get_most_similar_movie(description)
print("Next movie to watch:", most_similar_movie)