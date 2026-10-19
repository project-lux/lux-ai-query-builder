"""
Ask Gemini once for a two level topic list (disciplines -> specific topics) used to
seed generate-inputs.py. Review and edit the output by hand before generating inputs.

Usage: python generate-topics.py [--disciplines 30] [--topics 15] [--out generation-topics.json]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

import gemini_usage

with open("google_project.txt") as fh:
    gcp_name = fh.read().strip()

gemini_model = "gemini-3.8-flash"
gemini = genai.Client(vertexai=True, project=gcp_name, location="global")

collections_blurb = """LUX is Yale University's unified collections discovery platform. It covers
Yale's libraries (Sterling, Beinecke Rare Book & Manuscript Library, Lewis Walpole, music,
law, medical and divinity libraries), the Yale University Art Gallery, the Yale Center for
British Art, the Peabody Museum of Natural History, the Yale Collection of Historical
Musical Instruments, and Yale's archives. Records include books, manuscripts, artworks,
photographs, maps, natural history specimens, archaeological and ethnographic objects,
archival collections, people, organizations, places, concepts and events, from all periods
and all parts of the world.

The libraries hold millions of modern books, journals, theses, datasets and textbooks
across every field of research, so current science, medicine, engineering and technology
are just as important as history and the arts."""

required_disciplines = [
    "Physics", "Quantum Science and Technology", "Astronomy and Astrophysics",
    "Chemistry", "Mathematics and Statistics", "Computer Science and Artificial Intelligence",
    "Engineering", "Molecular and Cell Biology", "Genetics and Genomics", "Neuroscience",
    "Ecology and Evolutionary Biology", "Medicine and Clinical Research", "Public Health",
    "Nursing", "Pharmacology", "Psychology", "Earth and Climate Science",
    "Materials Science", "Architecture and Urban Planning", "Economics",
]

discipline_prompt = """{blurb}

The following disciplines are already on the list:
{required}

List {n} further broad disciplines or areas of study that university students would draw
on when searching these collections, to complete the coverage: the arts, humanities,
social sciences, law, Yale history, and anything else missing. Do not repeat or overlap
with the disciplines already listed.

Return a JSON array of {n} short discipline names."""

topic_prompt = """{blurb}

The full list of disciplines being covered is:
{disciplines}

For the discipline "{discipline}", list {n} specific topics that a student might search
the Yale collections for. Topics should be concrete and varied: periods, regions,
movements, genres, materials, types of object or document, notable people or groups,
events, scientific groups of organisms or minerals, and so on. Mix well known topics with
less obvious ones, and include topics from many parts of the world, not just Europe and
North America. For science, medicine, engineering and other research fields, most topics
should be current or recent research areas, methods and textbook subjects (for example
"quantum error correction" or "CRISPR gene editing in crops"), with only a few about the
history of the field. Do not tie topics to specific named collections or archives. Do
not repeat topics that belong more naturally to another discipline in the list.

Each topic should be a short phrase (2 to 8 words). Return a JSON array of {n} strings."""


def ask_list(prompt):
    cfg = types.GenerateContentConfig(
        temperature=1.0,
        response_mime_type="application/json",
        response_schema={"type": "ARRAY", "items": {"type": "STRING"}},
    )
    gemini_usage.check()
    resp = gemini.models.generate_content(
        model=gemini_model, contents=prompt, config=cfg
    )
    gemini_usage.record(resp.usage_metadata, "generate-topics")
    return json.loads(resp.text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disciplines", type=int, default=20, help="in addition to required")
    ap.add_argument("--topics", type=int, default=15)
    ap.add_argument("--out", default="generation-topics.json")
    args = ap.parse_args()

    required = "\n".join(f"  * {d}" for d in required_disciplines)
    disciplines = required_disciplines + ask_list(
        discipline_prompt.format(
            blurb=collections_blurb, required=required, n=args.disciplines
        )
    )
    print(f"{len(disciplines)} disciplines")
    dlist = "\n".join(f"  * {d}" for d in disciplines)

    def topics_for(d):
        return ask_list(
            topic_prompt.format(
                blurb=collections_blurb, disciplines=dlist, discipline=d, n=args.topics
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        all_topics = list(pool.map(topics_for, disciplines))

    out = [{"discipline": d, "topics": t} for d, t in zip(disciplines, all_topics)]
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {sum(len(t) for t in all_topics)} topics to {args.out}; "
          f"Gemini spend so far ${gemini_usage.spent():.4f}")


if __name__ == "__main__":
    main()
