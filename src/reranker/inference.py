import gzip
import string
from typing import List, Tuple, Dict

from bioc import biocxml
from swift.llm import InferRequest, RequestConfig
from tqdm import tqdm

MAX_NEW_TOKENS = 20
TEMPERATURE = 0


def _build_request(sentence_text: str, entity_text: str, candidates: List[str]):
    """
    candidates: list of "id>name" strings, e.g. ["D001>Influenza", "D002>Flu shot"]
    Returns (InferRequest, lookup_table, none_letter)
    """
    candidates = candidates[:20]
    num_candidates = len(candidates)
    letters = list(string.ascii_uppercase[:num_candidates + 1])
    candidate_letters = letters[:-1]
    none_letter = letters[-1]

    lookup = {}
    for i, c_str in enumerate(candidates):
        c_id, c_name = c_str.split(">")
        lookup[candidate_letters[i]] = {"id": c_id, "name": c_name}

    options = "\n".join(f"{l}: {data['name']}" for l, data in lookup.items())
    options += f"\n{none_letter}: None of the above."

    user_content = (
        f"<Instruct>: Given the context '{sentence_text}', "
        f"select the correct biomedical concept corresponding "
        f"to '{entity_text}'. Answer using one of the provided options.\n"
        f"<Options>: {options}"
    )

    request = InferRequest(messages=[{"role": "user", "content": user_content}])
    return request, lookup, none_letter


def _parse_response(response_text: str, lookup: Dict, none_letter: str) -> str:
    pred_option = response_text.lower().split("answer")[-1].replace(":", "").strip()
    pred_letter = pred_option[0].upper() if pred_option else ""

    if pred_letter == none_letter:
        return "None of the above"
    elif pred_letter in lookup:
        return "||".join([lookup[pred_letter]['id'], lookup[pred_letter]['name']])
    else:
        return pred_option  # fallback if the model hallucinates a letter


def rerank(model, items: List[Tuple[str, str, List[str]]]) -> List[str]:
    """
    Generic reranking over a list of (sentence_text, entity_text, candidates) tuples.
    Returns a list of prediction strings, one per item, in order.

    `model` is a BiolinkQwenModel (see model.py).
    """
    requests = []
    lookups = []

    for sentence_text, entity_text, candidates in items:
        request, lookup, none_letter = _build_request(sentence_text, entity_text, candidates)
        requests.append(request)
        lookups.append((lookup, none_letter))

    if not requests:
        return []

    request_config = RequestConfig(max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    resp_list = model.engine.infer(requests, request_config)

    results = []
    for resp, (lookup, none_letter) in zip(resp_list, lookups):
        response_text = resp.choices[0].message.content
        results.append(_parse_response(response_text, lookup, none_letter))
    return results


def rerank_biocxml_file(model, xml_path: str) -> None:
    """
    Drop-in equivalent of the original script's rerank_candidates():
    reads a BioC XML (optionally .gz) file, reranks every annotation's
    candidates in place, and overwrites the file.
    """
    is_gz = str(xml_path).endswith('.gz')
    opener = gzip.open if is_gz else open

    with opener(xml_path, 'rt', encoding='utf-8') as fp:
        collection = biocxml.load(fp)

    tasks = []
    items = []

    for doc in tqdm(collection.documents, desc="Reranking documents"):
        for passage in doc.passages:
            for sent in passage.sentences:
                for anno in sent.annotations:
                    candidates_str = anno.infons.get('candidates', "")
                    if not candidates_str:
                        continue
                    candidates = candidates_str.replace("||", ">").split("|")
                    tasks.append(anno)
                    items.append((sent.text, anno.text, candidates))

    predictions = rerank(model, items)

    for anno, pred in zip(tasks, predictions):
        anno.infons['reranker'] = pred

    write_mode = 'wt' if is_gz else 'w'
    with opener(xml_path, write_mode, encoding='utf-8') as fp:
        biocxml.dump(collection, fp)

    print(f"Finished processing {len(items)} annotations in {xml_path}")
