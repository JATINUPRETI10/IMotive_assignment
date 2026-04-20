import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# loading small embedding model (fast enough for testing)
model = SentenceTransformer('all-MiniLM-L6-v2')


# persona descriptions (given in assignment)
bot_profiles = {
    "A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    
    "B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    
    "C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}


# preparing data for vector search
ids = list(bot_profiles.keys())
profiles = list(bot_profiles.values())

# converting text to vectors
profile_vectors = model.encode(profiles, normalize_embeddings=True)

# FAISS setup
vec_size = profile_vectors.shape[1]
faiss_index = faiss.IndexFlatIP(vec_size)
faiss_index.add(np.array(profile_vectors))


def pick_bot_for_post(text, threshold=0.5):
    """
    simple matching logic to decide which bot should respond
    """

    # quick check to avoid weird input issues
    if not text or text.strip() == "":
        return {"error": "no input text"}

    # embed incoming text
    text_vec = model.encode([text], normalize_embeddings=True)

    # search across all stored profiles
    sim, idx = faiss_index.search(np.array(text_vec), k=len(bot_profiles))

    sim_scores = sim[0]
    indices = idx[0]

    matches = []
    seen_ids = set()

    # building match list
    for score, i in zip(sim_scores, indices):
        bot = ids[i]

        if bot not in seen_ids:
            matches.append({
                "bot_id": bot,
                "score": float(score)
            })
            seen_ids.add(bot)

    # sort highest first
    matches.sort(key=lambda x: x["score"], reverse=True)

    top_match = matches[0]

    # fallback if similarity is low
    # picked 0.5 after trying a few sample posts
    if top_match["score"] < threshold:
        chosen_bot = "A"
    else:
        chosen_bot = top_match["bot_id"]

    return {
        "bot_id": chosen_bot,
        "similarity": top_match["score"],
        "all_matches": matches,
        "persona": bot_profiles[chosen_bot]
    }


# quick manual test
if __name__ == "__main__":
    sample = "Stock market crash is affecting investors globally"

    res = pick_bot_for_post(sample)

    print("Chosen Bot:", res["bot_id"])
    print("Similarity:", res["similarity"])
    print("Matches:", res["all_matches"])