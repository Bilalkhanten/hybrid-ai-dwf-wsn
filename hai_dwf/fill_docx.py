"""Write the aggregated results into the revised manuscript (.docx), cell by cell, keeping the yellow highlight.

    python -m hai_dwf.fill_docx --docx IASC_-Adnan_revised.docx --patches results/patches.json --out IASC_-Adnan_final.docx

Only cells that still contain TBR / TBC are overwritten (use --force to overwrite measured cells too).  The placeholder
caption note under each table ("TBR = to be reported ...") is replaced by the note generated in aggregate.py.
Cells that remain TBR/TBC are listed at the end (they have no data behind them).
"""
import argparse, copy, json, re, shutil, tempfile, zipfile
from pathlib import Path
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
q = lambda t: f"{{{W}}}{t}"
SIG = {"T15": ("Variant", "Holm-adj. p"), "T16": ("Entropy regime (nats)", "Dominance hypothesis"), "T17": ("Dataset", "Original single run"),
       "T18": ("Setting", "DWF parameters tuned on"), "T19": ("Attack", "Threat model"), "T20": ("Stage", "Latency M55 / U74"),
       "A2": ("Component", "Provenance"), "A3": ("Item", "Value / setting"),
       "T21": ("Strategy", "Macro-F1"), "T22": ("Component (method)", "Top-3"), "T23": ("Configuration", "Dataset")}


def text(e): return "".join(t.text or "" for t in e.iter(q("t")))


def find_tables(body):
    found = {}
    for tbl in body.iter(q("tbl")):
        rows = tbl.findall(q("tr"))
        if not rows:
            continue
        hdr = [text(c) for c in rows[0].findall(q("tc"))]
        h = " | ".join(hdr)
        for k, (a, b) in SIG.items():
            if hdr and hdr[0].startswith(a) and b in h:
                found[k] = tbl
    return found


def highlight(run):
    rpr = run.find(q("rPr"))
    if rpr is None:
        rpr = etree.SubElement(run, q("rPr")); run.insert(0, rpr)
    if rpr.find(q("highlight")) is None:
        hl = etree.Element(q("highlight")); hl.set(q("val"), "yellow")
        # schema order: highlight comes after color/sz/szCs; append at a safe position (before u/effect/bdr/shd/...)
        after = ("rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline", "shadow", "emboss", "imprint",
                 "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs")
        pos = 0
        for i, ch in enumerate(rpr):
            if etree.QName(ch).localname in after:
                pos = i + 1
        rpr.insert(pos, hl)


def set_text(container, new):
    """container = w:tc or w:p; put `new` in the first text run of the first paragraph and blank the rest."""
    p = container if container.tag == q("p") else container.find(q("p"))
    runs = [r for r in p.iter(q("r")) if r.find(q("t")) is not None]
    if not runs:
        r = etree.SubElement(p, q("r")); t = etree.SubElement(r, q("t")); runs = [r]
    first = runs[0]; t = first.find(q("t")); t.text = new; t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for r in runs[1:]:
        p_ = r.getparent(); p_.remove(r)
    for extra in container.findall(q("p"))[1:] if container.tag == q("tc") else []:
        container.remove(extra)
    highlight(first)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docx", required=True); ap.add_argument("--patches", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    patches = json.load(open(a.patches))
    tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(a.docx).extractall(tmp)
    tree = etree.parse(str(tmp / "word" / "document.xml")); body = tree.getroot().find(q("body"))
    tabs = find_tables(body); done = missing = 0
    for p in patches:
        tbl = tabs.get(p["table"])
        if tbl is None:
            print(f"[warn] table {p['table']} not found"); continue
        if "note" in p:
            nxt = tbl.getnext()
            if nxt is not None and nxt.tag == q("p") and re.match(r"\s*(TBR|TBC)", text(nxt)):
                set_text(nxt, p["note"])
            continue
        rows = tbl.findall(q("tr"))
        key = p["row"]; hit = None
        for tr in rows[1:]:
            cells = tr.findall(q("tc")); first = text(cells[0])
            lab = f"{first}|{text(cells[1])}" if p["table"] in ("T17", "T18", "T23") else first
            if lab.startswith(key) if p["table"] not in ("T17", "T18", "T23") else lab == key or lab.startswith(key):
                hit = cells; break
        if hit is None:
            print(f"[warn] row '{key}' not found in {p['table']}"); missing += 1; continue
        cell = hit[p["col"]]; cur = text(cell)
        if a.force or re.search(r"\b(TBR|TBC)\b", cur):
            if re.search(r"\(Table 13\)", cur) and "TBR" in cur:       # partially filled cell: keep the known half
                set_text(cell, p["text"] + "  [Table 13 value superseded by the measurement]")
            else:
                set_text(cell, p["text"])
            done += 1
    tree.write(str(tmp / "word" / "document.xml"), xml_declaration=True, encoding="UTF-8", standalone=True)
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        ct = tmp / "[Content_Types].xml"; z.write(ct, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ct:
                z.write(f, f.relative_to(tmp).as_posix())
    left = []
    for k, tbl in find_tables(etree.parse(str(tmp / "word" / "document.xml")).getroot().find(q("body"))).items():
        for tr in tbl.findall(q("tr"))[1:]:
            cs = tr.findall(q("tc"))
            for j, c in enumerate(cs):
                if re.search(r"\b(TBR|TBC)\b", text(c)):
                    left.append(f"{k} | {text(cs[0])[:50]} | col {j}")
    print(f"filled {done} cells, {missing} rows not found, {len(left)} cells still TBR/TBC")
    for l in left[:80]:
        print("  ", l)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
