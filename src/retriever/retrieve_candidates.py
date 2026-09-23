import numpy as np
import gzip, json, torch, argparse
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from dense_vectors import make_dense_lookup


def main():
    parser = argparse.ArgumentParser('Retrieve candidates and save them directly into BioC XML')
    parser.add_argument('--input', required=True, type=str, help='json with queries')
    parser.add_argument('--kb_vectors', required=True, type=str, help='Precalculated vectors')
    parser.add_argument('--kb_mapping', required=True, type=str, help='Precalculated vectors')
    parser.add_argument('--top_k', required=False, type=int, default=10, help='Max candidates')
    parser.add_argument('--model_name', required=True, type=str, help='Transformer model')
    parser.add_argument('--output_file', required=True, type=str, help='Output Path for the modified BioC XML (gzipped)')
    parser.add_argument('--apply_grf', action="store_true")
    parser.add_argument('--use_rocchio', action="store_true")
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--obo_prefix', type=str) #defaults to None
    args = parser.parse_args()

    # 1. Load the list of query records
    with open(args.input, "r", encoding='utf-8') as fp:
        records = json.load(fp)

    # 2. Build a deduplicated {entity_mention: {}} lookup dict from every
    #    record's "entities" list, since make_dense_lookup expects entities
    #    keyed by mention text (not nested inside records).
    unique_entities = {}
    
    for cui, e in records.items():
        unique_entities[e["label"]] = {
            "cui":cui, 
            "docid":e["document_id"],  
            "entity_text":e["label"], 
            "feedback":e["standard_name"]
        }

    print(f"Loading {len(unique_entities)} unique entity mentions "
          f"from {len(records)} queries for prefix {args.obo_prefix}.")

    # 3. Setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # 4. Load Vectors & Search
    print("Loading ontology vectors and performing search...")
    onto_vectors = np.load(args.kb_vectors)

    lookup = make_dense_lookup(
        model, tokenizer, onto_vectors, unique_entities,
        args.top_k, with_grf=args.apply_grf,
        use_rocchio=args.use_rocchio, alpha=args.alpha
    )
    print(f"Retrieved candidates for {len(lookup)} unique entity mentions.")

    # 4.5 load ontology id to text mapping
    mapping_path = Path(args.kb_vectors).parent / args.kb_mapping #"ontology_mapping.txt"

    with open(mapping_path, "r", encoding="utf-8") as f:
        ontology_entries = [l.strip() for l in f if l.strip()]

    print(len(ontology_entries), len(onto_vectors))
    assert len(ontology_entries) == len(onto_vectors)

    # 5. Resolve pipe-separated indices into candidate name strings, once
    #    per unique entity mention.
    entity_to_candidates = {}
    for mention in unique_entities:
        if mention in lookup:
            indices = [int(idx) for idx in lookup[mention].split("|")]
            entity_to_candidates[mention] = "|".join(
                ontology_entries[idx] for idx in indices
            )

    # 6. Inject candidates back into each record, aligned with its
    #    "entities" list (one candidate string per entity, in order).
    print("Injecting candidates into query records...")
    for cui, record in records.items():
        entity = record.get("label", "")
        record["candidates"] =  entity_to_candidates.get(entity, "") 

    # 7. Save the enriched records as a JSON array
    with open(args.output_file, "w", encoding='utf-8') as fp:
        json.dump(records, fp, indent=4)


if __name__ == "__main__":
    main()
