"""Document normalization for ATHENA (Milestone M2).

Accepts retrieved scientific literature (Paper models or raw metadata dicts)
and constructs a canonical NormalizedDocument preserving provenance, identity,
and access status.
"""

from typing import Any

from app.evidence.cleaner import clean_scientific_text
from app.evidence.models import NormalizedDocument
from app.retrieval.models import OpenAccessInfo, Paper


def normalize_paper(data: Paper | dict[str, Any]) -> NormalizedDocument:
    """Normalize a Paper model or paper dictionary into a canonical NormalizedDocument.

    Handles missing fields gracefully, preserves paper identity and URLs,
    and applies scientific text cleaning to titles and abstracts.

    Args:
        data: A Paper dataclass instance or a dictionary of paper metadata.

    Returns:
        A canonical NormalizedDocument instance.
    """
    if isinstance(data, Paper):
        paper_dict = data.to_dict()
    elif isinstance(data, dict):
        paper_dict = dict(data)
    else:
        raise TypeError(f"Expected Paper or dict, got {type(data).__name__}")

    raw_id = paper_dict.get("openalex_id") or paper_dict.get("id") or paper_dict.get("document_id") or ""
    document_id = str(raw_id).strip()

    # Clean title
    raw_title = paper_dict.get("title")
    title = clean_scientific_text(raw_title) if raw_title else None
    if title == "":
        title = None

    # Authors
    raw_authors = paper_dict.get("authors") or []
    authors: list[str] = []
    if isinstance(raw_authors, (list, tuple)):
        for author in raw_authors:
            if isinstance(author, str) and author.strip():
                authors.append(author.strip())
            elif isinstance(author, dict) and "display_name" in author:
                name = author.get("display_name")
                if isinstance(name, str) and name.strip():
                    authors.append(name.strip())

    # Publication Year
    raw_year = paper_dict.get("publication_year") or paper_dict.get("year")
    publication_year: int | None = None
    if raw_year is not None:
        try:
            publication_year = int(raw_year)
        except (ValueError, TypeError):
            publication_year = None

    # DOI
    raw_doi = paper_dict.get("doi")
    doi = str(raw_doi).strip() if isinstance(raw_doi, str) and raw_doi.strip() else None

    # Venue
    raw_venue = paper_dict.get("venue")
    venue = clean_scientific_text(raw_venue) if isinstance(raw_venue, str) and raw_venue.strip() else None
    if venue == "":
        venue = None

    # Landing page URL
    raw_url = paper_dict.get("landing_page_url") or paper_dict.get("url")
    landing_page_url = str(raw_url).strip() if isinstance(raw_url, str) and raw_url.strip() else None
    if landing_page_url is None:
        landing_page_url = doi or (document_id if document_id.startswith("http") else None)

    # Abstract text
    raw_abstract = paper_dict.get("abstract") or paper_dict.get("abstract_text")
    abstract_text = clean_scientific_text(raw_abstract) if raw_abstract else None
    if abstract_text == "":
        abstract_text = None

    # Sections (for future full text or structured documents)
    sections_raw = paper_dict.get("sections")
    sections: dict[str, str] = {}
    if isinstance(sections_raw, dict):
        for sec_name, sec_text in sections_raw.items():
            if isinstance(sec_name, str) and sec_text:
                cleaned_sec = clean_scientific_text(sec_text)
                if cleaned_sec:
                    sections[sec_name] = cleaned_sec

    # Cited by count
    raw_citations = paper_dict.get("cited_by_count") or paper_dict.get("citations")
    cited_by_count: int | None = None
    if raw_citations is not None:
        try:
            cited_by_count = int(raw_citations)
        except (ValueError, TypeError):
            cited_by_count = None

    # Open Access info
    oa_data = paper_dict.get("open_access")
    is_open_access: bool | None = None
    oa_status: str | None = None
    oa_url: str | None = None

    if isinstance(oa_data, OpenAccessInfo):
        is_open_access = oa_data.is_oa
        oa_status = oa_data.oa_status
        oa_url = oa_data.oa_url
    elif isinstance(oa_data, dict):
        is_open_access = oa_data.get("is_oa") if isinstance(oa_data.get("is_oa"), bool) else None
        oa_status = str(oa_data.get("oa_status")) if oa_data.get("oa_status") is not None else None
        oa_url = str(oa_data.get("oa_url")) if oa_data.get("oa_url") is not None else None

    source_db = str(paper_dict.get("source") or paper_dict.get("source_database") or "OpenAlex")

    return NormalizedDocument(
        document_id=document_id,
        title=title,
        authors=authors,
        publication_year=publication_year,
        doi=doi,
        venue=venue,
        landing_page_url=landing_page_url,
        abstract_text=abstract_text,
        sections=sections,
        cited_by_count=cited_by_count,
        is_open_access=is_open_access,
        oa_status=oa_status,
        oa_url=oa_url,
        source_database=source_db,
        raw_metadata=paper_dict,
    )
