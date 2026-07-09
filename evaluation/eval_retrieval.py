# evaluation/eval_retrieval.py
#
# Measures retrieval quality for rag_chat.py: for a fixed set of past
# conversation turns and test queries with a known "correct" turn, does
# retrieval actually surface that turn in its top-k?
#
# This compares two retrieval strategies head to head:
#   - dense only   (FAISS embedding similarity, what a minimal RAG demo uses)
#   - hybrid (RRF) (dense + BM25 keyword search, what rag_chat.py actually
#                    uses in the app)
#
# The test set is deliberately split into two query categories to make the
# comparison meaningful rather than just "hybrid usually wins":
#   - "semantic"      queries are paraphrases with no shared vocabulary with
#                      the source turn -- dense embeddings should carry these.
#   - "exact_keyword"  queries hinge on a specific name/ID/number a general
#                      embedding model has little reason to weight heavily --
#                      BM25 (and therefore hybrid) should carry these.
#
# Run with: python evaluation/eval_retrieval.py
# Needs the sentence-transformers model downloaded (first run needs internet).

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import rag_chat  # noqa: E402

# ----------------------------------------------------------------------
# Redirect rag_chat's persistence to a throwaway directory for this eval
# run, so it doesn't read or write the app's real conversation history.
# ----------------------------------------------------------------------
_tmp_dir = tempfile.mkdtemp(prefix="smartspeak_eval_")
rag_chat._DATA_DIR = _tmp_dir
rag_chat._INDEX_PATH = os.path.join(_tmp_dir, "faiss.index")
rag_chat._TEXTS_PATH = os.path.join(_tmp_dir, "texts.json")
rag_chat._loaded_from_disk = True  # skip disk load attempt; start from empty state


# ----------------------------------------------------------------------
# Test corpus: synthetic past conversation turns to index.
#
# Two attempts at this corpus already came back 100%/100% for both dense
# and hybrid retrieval -- meaning the test wasn't actually hard enough to
# tell the two methods apart. A benchmark where every method scores
# identically isn't validating anything, so this version leans harder into
# near-identical text blocks that differ *only* by an ID or name (four
# variants per group, not three) -- the case where dense embeddings
# genuinely struggle, since the sentence vectors end up nearly identical,
# and BM25's exact-token match should have a real, visible edge.
# ----------------------------------------------------------------------
CORPUS = [
    # -- travel: four semantically similar "weekend trip" turns (control group) --
    ("What's a good weekend trip near Bangalore?", "Coorg is a popular pick -- coffee estates and cool weather, about 5 hours by road."),
    ("Where can I go for a quick 2-day getaway from Bangalore?", "Chikmagalur has good hill views and is closer, about 4 hours away."),
    ("Suggest a relaxing short trip from Bangalore.", "Munnar in Kerala is a bit further but worth it for the tea plantations."),
    ("Any hill station near Bangalore for a family trip?", "Ooty is family-friendly with a toy train and botanical gardens, about 6 hours away."),

    # -- IT tickets: four near-identical text blocks, differ only by ticket number --
    ("Any update on ticket #48213?", "Ticket #48213: office printer offline in room 204. IT confirmed the issue and will send someone by end of day."),
    ("Any update on ticket #48217?", "Ticket #48217: office printer offline in room 204. This was resolved this morning and the printer is back online."),
    ("Any update on ticket #48221?", "Ticket #48221: office printer offline in room 204. Still open, waiting on a replacement toner cartridge."),
    ("Any update on ticket #48226?", "Ticket #48226: office printer offline in room 204. Escalated to a vendor technician, visit scheduled for tomorrow."),

    # -- invoices: four near-identical text blocks, differ only by invoice number --
    ("Has INV-88214 shipped?", "INV-88214: order shipped, arriving within 3 business days."),
    ("Has INV-88219 shipped?", "INV-88219: order shipped, arriving within 3 business days."),
    ("Has INV-77310 shipped?", "INV-77310: order shipped, arriving within 3 business days."),
    ("Has INV-90042 shipped?", "INV-90042: order shipped, arriving within 3 business days."),

    # -- pets: four near-identical text blocks, differ only by pet name + reason --
    ("Is Biscuit due for anything at his next vet visit?", "Biscuit (golden retriever) is due for a rabies booster at his next vet visit."),
    ("Is Mochi due for anything at her next vet visit?", "Mochi (tabby cat) is due for a dental cleaning at her next vet visit."),
    ("Is Leo due for anything at his next vet visit?", "Leo (beagle) is due for an ear infection check at his next vet visit."),
    ("Is Pepper due for anything at her next vet visit?", "Pepper (rabbit) is due for a nail trim at her next vet visit."),

    # -- adversarial semantic case: query shares zero literal tokens with the
    # correct turn, but shares a token ("car") with an unrelated distractor.
    # BM25 alone would likely favor the wrong document here; a properly
    # weighted hybrid should still lean on dense's correct semantic read.
    ("Is my Honda Civic ready to drive long distance?", "The Honda Civic we test drove needs new brake pads before it's roadworthy."),
    ("What's the best budget car insurance in Bangalore?", "HDFC Ergo and Digit both have competitively priced comprehensive car insurance plans."),
    ("My motorcycle's chain feels loose lately.", "My motorcycle's chain needs lubrication and a tension check every 500km."),

    # -- standalone topics: no close distractor, control group --
    ("Can you recommend a laptop for video editing?", "Look for at least 32GB RAM and a dedicated GPU -- the ASUS ROG Zephyrus G14 is a solid mid-range option."),
    ("What's a good book to read on a long flight?", "Project Hail Mary is a great pick for a long flight -- fast-paced and hard to put down."),
    ("How do I make my sourdough starter more active?", "Feed it 1:1:1 (starter:flour:water) every 12 hours at room temperature, and keep it away from drafts."),
]

# ----------------------------------------------------------------------
# Test queries: (query, expected_corpus_index, category)
# Indices below refer to CORPUS as ordered above (0-indexed).
# ----------------------------------------------------------------------
QUERIES = [
    ("Where should I go for a short getaway from Bangalore with cooler weather and coffee?", 0, "semantic"),
    ("Which nearby hill station has a toy train?", 3, "semantic"),
    ("Any update on ticket #48217?", 5, "exact_keyword"),
    ("Is ticket 48221 still open or resolved?", 6, "exact_keyword"),
    ("Has INV-88219 shipped out yet?", 9, "exact_keyword"),
    ("What's the status of INV-90042?", 11, "exact_keyword"),
    ("Does Biscuit need anything at his checkup?", 12, "exact_keyword"),
    ("What's Pepper due for at her next visit?", 15, "exact_keyword"),
    ("Is that used sedan okay to drive without new brakes first?", 16, "semantic"),
    ("Need a machine for editing 4K footage, any suggestions?", 19, "semantic"),
    ("Anything good to read on a long-haul flight?", 20, "semantic"),
]

K = 1  # strict: only the single top result counts as a hit


def dense_only_retrieve(query: str, k: int) -> list:
    indices = rag_chat._dense_rank(query, k)
    return [rag_chat._texts[i] for i in indices]


def run_eval():
    print(f"Indexing {len(CORPUS)} corpus turns...\n")
    for q, a in CORPUS:
        rag_chat.add_turn(q, a)

    results = {"dense": {"semantic": [], "exact_keyword": []}, "hybrid": {"semantic": [], "exact_keyword": []}}

    print(f"{'Query':45s} {'Category':14s} {'Dense hit@'+str(K):>11s} {'Hybrid hit@'+str(K):>12s}")
    print("-" * 86)

    for query, expected_idx, category in QUERIES:
        expected_q, expected_a = CORPUS[expected_idx]
        expected_text = f"Q: {expected_q}\nA: {expected_a}"

        dense_results = dense_only_retrieve(query, K)
        hybrid_results = rag_chat.retrieve_context(query, K)

        dense_hit = expected_text in dense_results
        hybrid_hit = expected_text in hybrid_results

        results["dense"][category].append(dense_hit)
        results["hybrid"][category].append(hybrid_hit)

        print(f"{query[:44]:45s} {category:14s} {str(dense_hit):>11s} {str(hybrid_hit):>12s}")

    print("\n" + "=" * 86)
    print("Hit-rate@%d by category" % K)
    print("=" * 86)
    for category in ["semantic", "exact_keyword"]:
        dense_rate = sum(results["dense"][category]) / len(results["dense"][category])
        hybrid_rate = sum(results["hybrid"][category]) / len(results["hybrid"][category])
        print(f"  {category:14s}  dense={dense_rate:.0%}   hybrid={hybrid_rate:.0%}")

    all_dense = results["dense"]["semantic"] + results["dense"]["exact_keyword"]
    all_hybrid = results["hybrid"]["semantic"] + results["hybrid"]["exact_keyword"]
    print(f"  {'overall':14s}  dense={sum(all_dense)/len(all_dense):.0%}   hybrid={sum(all_hybrid)/len(all_hybrid):.0%}")
    print()


if __name__ == "__main__":
    run_eval()