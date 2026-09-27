"""Convert the frozen Markdown result tables into LaTeX tables for the Supplementary Material, copying every cell.

    python tools/md2tex_tables.py spec.json out.tex

spec.json is a list of {"file", "heading", "index" (the n-th table under that heading, default 0), "caption",
"label", "align" (optional LaTeX column spec), "size" (optional font command, default \\footnotesize),
"drop_cols" (optional list of column indices to omit), "rename" (optional map from a whole cell, header cells included, to its paper label)}. The heading is matched as a prefix of a Markdown heading line.
Numbers are copied as text; only typography changes (unicode symbols to LaTeX, the minus sign, bold, dollars,
per cent). A table with more than 25 rows becomes a longtable.
"""
import json
import pathlib
import re
import sys

BS = chr(92)
UNI = {"Δ": "$" + BS + "Delta$", "σ": "$" + BS + "sigma$", "τ": "$" + BS + "tau$", "ε": "$" + BS + "varepsilon$",
       "−": "$-$", "×": "$" + BS + "times$", "≥": "$" + BS + "geq$", "≤": "$" + BS + "leq$", "→": "$" + BS + "to$",
       "·": "$" + BS + "cdot$", "±": "$" + BS + "pm$", "≈": "$" + BS + "approx$", "—": "--", "–": "--",
       "“": "``", "”": "''", "’": "'", "μ": "$" + BS + "mu$", "ₑ": "", "²": "$^2$", "₂": "$_2$", "°": "$^" + BS +
       "circ$", "…": BS + "ldots{}", "⁻": "$^{-}$", "¹": "$^1$", "⁰": "$^0$", "⁴": "$^4$", "⁵": "$^5$", "⁶": "$^6$",
       "⁹": "$^9$", "ᵃ": "$^a$", "ᵇ": "$^b$", "ᶜ": "$^c$", "✓": "$" + BS + "checkmark$", "◐": "$" + BS + "circ$",
       "∈": "$" + BS + "in$", "≠": "$" + BS + "neq$", "∞": "$" + BS + "infty$", "π": "$" + BS + "pi$",
       "Π": "$" + BS + "Pi$", "ℓ": "$" + BS + "ell$", "κ": "$" + BS + "kappa$", "η": "$" + BS + "eta$",
       "ρ": "$" + BS + "rho$", "β": "$" + BS + "beta$", "θ": "$" + BS + "theta$", "α": "$" + BS + "alpha$",
       "p" + chr(0x303): "$" + BS + "tilde{p}$", "⌈": "$" + BS + "lceil$", "⌉": "$" + BS + "rceil$",
       "§": BS + "S{}", "½": "$" + BS + "frac12$", "¢": BS + "textcent{}"}


def tex_cell(s):
    s = s.strip()
    s = re.sub(r"`([^`]*)`", r"\1", s)                               # code spans -> plain
    s = s.replace("Δ_info", "\x01")                                   # restored as math after escaping
    s = s.replace(BS, BS + "textbackslash{}")
    for ch in "&%#_{}":
        s = s.replace(ch, BS + ch)
    s = s.replace("$", BS + "$")
    s = s.replace("~", BS + "textasciitilde{}").replace("^", BS + "textasciicircum{}")
    for k, v in UNI.items():
        s = s.replace(k, v)
    s = re.sub(r"(?<=[A-Za-z0-9_])/(?=[A-Za-z])", lambda m: "/" + BS + "allowbreak{}", s)   # paths break at a slash
    s = re.sub(r"(?<![\w$])-(?=\d)", "$-$", s)                        # a leading minus sign before a number
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: BS + "textbf{" + m.group(1) + "}", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", lambda m: BS + "emph{" + m.group(1) + "}", s)
    return s.replace("\x01", "$" + BS + "Delta_{" + BS + "mathrm{info}}$")


def md_tables(text, heading):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("#") and l.lstrip("#").strip().startswith(heading)),
                 None)
    if start is None:
        raise KeyError(heading)
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    tables, cur = [], []
    for l in lines[start + 1:]:
        if l.startswith("#") and len(l) - len(l.lstrip("#")) <= level:
            break
        if l.strip().startswith("|"):
            cur.append(l.strip())
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def split_row(l):
    return [c for c in l.strip().strip("|").split("|")]


def fit_row(cells, n):
    """Some records write labels such as 'A+staged|spine' inside a cell, which splits it; merge the leading cells back."""
    extra = len(cells) - n
    if extra > 0:
        cells = [" / ".join(c.strip() for c in cells[:extra + 1])] + cells[extra + 1:]
    return cells


def wrap_head(c, width=14):
    """A long header cell becomes a small bottom-aligned stack of lines of at most `width` characters."""
    words, lines, cur = c.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    if len(lines) == 1:
        return tex_cell(c)
    inner = (BS + BS).join(tex_cell(l) for l in lines)
    return BS + "begin{tabular}[b]{@{}c@{}}" + inner + BS + "end{tabular}"


def to_latex(rows, caption, label, align=None, size=None, drop=(), rename=None, override=(), placement="!htbp"):
    n0 = len(split_row(rows[0]))
    head = [c for i, c in enumerate(split_row(rows[0])) if i not in drop]
    body = [[c for i, c in enumerate(fit_row(split_row(r), n0)) if i not in drop] for r in rows[2:]]
    head = [(rename or {}).get(c.strip(), c) for c in head]
    body = [[(rename or {}).get(c.strip(), c) for c in r] for r in body]
    for first, col, text in override:                                  # one cell in the row whose first cell is `first`
        for r in body:
            if r[0].strip() == first:
                r[col] = text
    n = len(head)
    align = align or ("l" + "r" * (n - 1))
    size = size if size is not None else BS + "footnotesize"
    long = len(body) > 25
    out = []
    hdr = " & ".join(wrap_head(c) for c in head) + " " + BS + BS
    if long:
        out += [BS + "begingroup" + size, BS + "setlength{" + BS + "tabcolsep}{3pt}",
                BS + "begin{longtable}{@{}" + align + "@{}}",
                BS + "caption{" + caption + "}" + BS + "label{" + label + "}" + BS + BS,
                BS + "toprule", hdr, BS + "midrule", BS + "endfirsthead",
                BS + "toprule", hdr, BS + "midrule", BS + "endhead", BS + "bottomrule", BS + "endfoot"]
        out += [" & ".join(tex_cell(c) for c in r) + " " + BS + BS for r in body]
        out += [BS + "end{longtable}", BS + "endgroup", ""]
    else:
        out += [BS + "begin{table}[" + placement + "]", BS + "centering", BS + "caption{" + caption + "}",
                BS + "label{" + label + "}", size, BS + "setlength{" + BS + "tabcolsep}{3pt}",
                BS + "begin{adjustbox}{max width=" + BS + "textwidth}",
                BS + "begin{tabular}{@{}" + align + "@{}}", BS + "toprule", hdr, BS + "midrule"]
        out += [" & ".join(tex_cell(c) for c in r) + " " + BS + BS for r in body]
        out += [BS + "bottomrule", BS + "end{tabular}", BS + "end{adjustbox}", BS + "end{table}", ""]
    return "\n".join(out)


def main(spec_path, out_path, placement="!htbp"):
    """Entries with a "group" are written to <out>_<group>.tex, one file per group in spec order, so each group can be
    input after the text it belongs to; an entry {"include": path} inputs a hand-built table at that position."""
    root = pathlib.Path(__file__).resolve().parents[1]
    spec = json.loads(pathlib.Path(spec_path).read_text(encoding="utf-8"))
    header = "% Generated by tools/md2tex_tables.py from the frozen Markdown tables; do not edit by hand."
    groups = {}
    for s in spec:
        g = s.get("group", "")
        chunks = groups.setdefault(g, [header])
        if "include" in s:
            chunks.append(BS + "input{" + s["include"] + "}\n")
            continue
        text = (root / s["file"]).read_text(encoding="utf-8")
        rows = md_tables(text, s["heading"])[s.get("index", 0)]
        src = f"% source: {s['file']} :: {s['heading']} [table {s.get('index', 0)}]"
        chunks.append(src + "\n" + to_latex(rows, s["caption"], s["label"], s.get("align"), s.get("size"),
                                            tuple(s.get("drop_cols", ())), s.get("rename"),
                                            tuple(tuple(o) for o in s.get("cell_override", ())), placement))
    out = pathlib.Path(out_path)
    for g, chunks in groups.items():
        path = out if g == "" else out.with_name(f"{out.stem}_{g}{out.suffix}")
        path.write_text("\n".join(chunks), encoding="utf-8")
        print(f"written {path}: {len(chunks) - 1} tables")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--placement=")]
    place = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--placement=")), "!htbp")
    main(args[0], args[1], place)
