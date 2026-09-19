#!/usr/bin/env python3
"""
LCCI Corporate Class Final Voters' List -> structured CSV.

Usage:
    python extract_voters.py input.pdf -o voters_2026.csv -r extraction_report.txt
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field

import pdfplumber

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

COLUMNS = [
    "vote_number", "membership_number", "member_name", "designation",
    "company_name", "business_address", "phone_numbers", "nic", "ntn", "gstn",
]

LINE_TOL = 3.0          # vertical tolerance (pt) when grouping words into lines
COLUMN_GAP_MIN = 12.0   # minimum white gutter width (pt) to accept a split

RE_VOTE = re.compile(r"^Vote\s*#\s*(\S+)\s+Membership\s*#\s*(\S+)\s*$", re.I)
RE_PHONE = re.compile(r"^Phone\s*:\s*(.*)$", re.I)
RE_NIC_NTN = re.compile(r"^NIC\s*:\s*(\S*)\s*(?:NTN\s*:\s*(\S*))?\s*$", re.I)
RE_NTN_ONLY = re.compile(r"^NTN\s*:\s*(\S*)\s*$", re.I)
RE_GSTN = re.compile(r"^GSTN\s*:\s*(\S*)\s*$", re.I)
RE_HEADER = re.compile(r"Corporate Class Final Voters", re.I)
RE_FOOTER = re.compile(r"©|Lahore Chamber of Commerce|^\d+\s*/\s*\d+$", re.I)

# name line: "MR. WALEED SANAULLAH - Director"  (split on the LAST " - ")
RE_NAME_DESIG = re.compile(r"^(?P<name>.+?)\s+-\s+(?P<desig>[^-]+)$")

# company-name continuation heuristics
RE_DANGLING = re.compile(r"(\(PVT\.?\)|\(SMC-PVT\.?\)|&|-|,|\(PRIVATE\))\s*$", re.I)
RE_SUFFIX_ONLY = re.compile(
    r"^(LTD\.?|LIMITED\.?|LTD\.?\s*\.?|PVT\.?\s*LTD\.?|"
    r"\(PVT\.?\)\s*LTD\.?|CO\.?|COMPANY|INDUSTRIES|ENTERPRISES)$", re.I)

TRUNCATION_MARKS = ("\u2026", "...")

# The sample row in the spec shows the name without the honorific. Source text is
# kept verbatim by default; flip this to True to strip the leading title instead.
STRIP_HONORIFICS = False
HONORIFICS = ("MR.", "MRS.", "MS.", "MISS", "DR.", "ENGR.", "SH.", "SYED",
              "MIAN", "MALIK", "RANA", "CH.", "CHAUDHRY", "HAJI", "PROF.")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------

@dataclass
class Record:
    vote_number: str = ""
    membership_number: str = ""
    member_name: str = ""
    designation: str = ""
    company_name: str = ""
    business_address: str = ""
    phone_numbers: str = ""
    nic: str = ""
    ntn: str = ""
    gstn: str = ""
    # provenance / QA (never written to the main CSV)
    page: int = 0
    column: str = ""
    raw_lines: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_row(self):
        return {c: getattr(self, c) for c in COLUMNS}


# ----------------------------------------------------------------------------
# Layout -> ordered text lines
# ----------------------------------------------------------------------------

def find_column_split(words, page_width):
    """Locate the widest vertical gutter near the page centre; None if 1 column."""
    xs = sorted((w["x0"], w["x1"]) for w in words)
    lo, hi = page_width * 0.35, page_width * 0.65
    best, best_gap = None, 0.0
    reach = None
    for x0, x1 in xs:
        if reach is not None and x0 - reach > best_gap and lo < (x0 + reach) / 2 < hi:
            best_gap, best = x0 - reach, (x0 + reach) / 2
        reach = x1 if reach is None else max(reach, x1)
    if best is None or best_gap < COLUMN_GAP_MIN:
        return None
    return best


def words_to_lines(words):
    """Group words into visual lines: [(text, [word, ...])], top-to-bottom."""
    lines, current, current_top = [], [], None
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if current_top is not None and abs(w["top"] - current_top) <= LINE_TOL:
            current.append(w)
        else:
            if current:
                lines.append(current)
            current, current_top = [w], w["top"]
    if current:
        lines.append(current)
    out = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w["x0"])
        text = re.sub(r"\s+", " ", " ".join(w["text"] for w in ln)).strip()
        if text:
            out.append((text, ln))
    return out


def page_lines(page):
    """Return [(line_text, column_label)] for one page, in reading order."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return []

    # 1) drop running header / footer lines before measuring the gutter
    body_words = []
    for text, ws in words_to_lines(words):
        if not _is_chrome(text):
            body_words.extend(ws)
    if not body_words:
        return []

    # 2) locate the gutter and split
    split = find_column_split(body_words, page.width)
    if split is None:
        return [(t, "single") for t, _ in words_to_lines(body_words)]

    left = [w for w in body_words if (w["x0"] + w["x1"]) / 2 < split]
    right = [w for w in body_words if (w["x0"] + w["x1"]) / 2 >= split]
    out = []
    for label, group in (("left", left), ("right", right)):
        out.extend((t, label) for t, _ in words_to_lines(group))
    return out


def _is_chrome(text):
    """Running header / footer / page number."""
    return bool(RE_HEADER.search(text) or (RE_FOOTER.search(text) and not RE_VOTE.match(text)))


# ----------------------------------------------------------------------------
# Record segmentation + field extraction
# ----------------------------------------------------------------------------

def segment_records(all_lines):
    """all_lines: [(text, page, column)] -> [Record] (fields still unparsed)."""
    records, cur = [], None
    orphan = []
    for text, pg, col in all_lines:
        m = RE_VOTE.match(text)
        if m:
            if cur:
                records.append(cur)
            cur = Record(vote_number=m.group(1), membership_number=m.group(2),
                         page=pg, column=col)
            cur.raw_lines.append(text)
        elif cur is not None:
            cur.raw_lines.append(text)
        else:
            orphan.append((text, pg, col))
    if cur:
        records.append(cur)
    return records, orphan


def parse_record(rec):
    """Map a record's raw lines onto the fixed schema."""
    body = rec.raw_lines[1:]          # drop the "Vote # ... Membership # ..." line
    block, company_addr = [], []

    i = 0
    # 1) name / designation
    if i < len(body):
        m = RE_NAME_DESIG.match(body[i])
        if m:
            rec.member_name = m.group("name").strip()
            if STRIP_HONORIFICS:
                for h in HONORIFICS:
                    if rec.member_name.upper().startswith(h + " "):
                        rec.member_name = rec.member_name[len(h):].strip()
                        break
            rec.designation = m.group("desig").strip()
            i += 1
        else:
            rec.warnings.append("name/designation line not recognised")
    else:
        rec.warnings.append("record body is empty")

    # 2) company + address, up to the first Phone/NIC/NTN/GSTN line
    while i < len(body) and not (RE_PHONE.match(body[i]) or RE_NIC_NTN.match(body[i])
                                 or RE_NTN_ONLY.match(body[i]) or RE_GSTN.match(body[i])):
        company_addr.append(body[i])
        i += 1

    # 3) trailing identifier lines
    while i < len(body):
        line = body[i]
        if (m := RE_PHONE.match(line)):
            phones = [p.strip() for p in m.group(1).split(",") if p.strip()]
            rec.phone_numbers = "; ".join(phones)
        elif (m := RE_NIC_NTN.match(line)):
            rec.nic = (m.group(1) or "").strip()
            if m.group(2):
                rec.ntn = m.group(2).strip()
        elif (m := RE_NTN_ONLY.match(line)):
            rec.ntn = m.group(1).strip()
        elif (m := RE_GSTN.match(line)):
            rec.gstn = m.group(1).strip()
        else:
            block.append(line)          # unexpected trailing text
        i += 1

    if block:
        rec.warnings.append("unrecognised trailing line(s): " + " | ".join(block))

    split_company_address(rec, company_addr)
    validate(rec)
    return rec


def split_company_address(rec, lines):
    """First line is the company; following lines join it only if it clearly wrapped."""
    if not lines:
        rec.warnings.append("no company/address lines found")
        return
    company = [lines[0]]
    idx = 1
    while idx < len(lines):
        prev, cand = company[-1], lines[idx]
        if RE_DANGLING.search(prev) and (RE_SUFFIX_ONLY.match(cand) or len(cand.split()) <= 2):
            company.append(cand)
            rec.warnings.append(f"company name treated as wrapped: {cand!r}")
            idx += 1
        else:
            break
    rec.company_name = " ".join(company)
    rec.business_address = " ".join(lines[idx:]).strip()
    if not rec.business_address:
        rec.warnings.append("no business address")


def validate(rec):
    if any(m in rec.business_address for m in TRUNCATION_MARKS):
        rec.warnings.append("address appears truncated in the source PDF")
    for f in ("member_name", "designation", "company_name", "phone_numbers", "nic", "ntn"):
        if not getattr(rec, f):
            rec.warnings.append(f"missing {f}")
    if rec.nic and not re.fullmatch(r"[\d\-]{13,20}", rec.nic):
        rec.warnings.append(f"NIC has unexpected format: {rec.nic}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def extract(pdf_path):
    all_lines, pages_without_text = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            lines = page_lines(page)
            if not lines:
                pages_without_text.append(n)     # likely needs OCR
                continue
            all_lines.extend((t, n, c) for t, c in lines)

    records, orphan = segment_records(all_lines)
    records = [parse_record(r) for r in records]
    return records, orphan, pages_without_text


def build_report(records, orphan, ocr_pages, pdf_path):
    L = [f"Extraction report for: {pdf_path}",
         f"Records extracted: {len(records)}", ""]

    if ocr_pages:
        L += [f"Pages with no extractable text (OCR required): {ocr_pages}", ""]
    if orphan:
        L += [f"Text before the first Vote # marker ({len(orphan)} line(s)):"]
        L += [f"  p{p}/{c}: {t}" for t, p, c in orphan[:20]] + [""]

    seen, dupes = {}, []
    for r in records:
        seen.setdefault(r.vote_number, []).append(r)
    for v, rs in seen.items():
        if len(rs) > 1:
            dupes.append((v, [(r.page, r.column) for r in rs]))
    L += [f"Duplicate vote numbers: {dupes if dupes else 'none'}"]

    nums = sorted(int(v) for v in seen if v.isdigit())
    gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in set(nums)] if nums else []
    L += [f"Vote number range: {nums[0]:04d}-{nums[-1]:04d}" if nums else "Vote number range: n/a",
          f"Sequence gaps: {[f'{g:04d}' for g in gaps] if gaps else 'none'}", ""]

    field_missing = {c: sum(1 for r in records if not getattr(r, c)) for c in COLUMNS}
    L += ["Empty-field counts:"]
    L += [f"  {c:<20} {n}" for c, n in field_missing.items()] + [""]

    flagged = [r for r in records if r.warnings]
    L += [f"Records requiring review: {len(flagged)}"]
    for r in flagged:
        L.append(f"  [{r.vote_number}] p{r.page}/{r.column}: " + "; ".join(r.warnings))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="voters_2026.csv")
    ap.add_argument("-r", "--report", default="extraction_report.txt")
    args = ap.parse_args()

    records, orphan, ocr_pages = extract(args.pdf)
    if not records:
        sys.exit("No records found - check the layout assumptions or run OCR first.")

    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
        w.writeheader()
        for r in records:
            w.writerow(r.as_row())

    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(build_report(records, orphan, ocr_pages, args.pdf))

    print(f"{len(records)} records -> {args.out}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
