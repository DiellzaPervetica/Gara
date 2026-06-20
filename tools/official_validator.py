"""Independent validator implementing the README / netlify official semantics.

Time is computed from the ACTUAL submitted node list (sum of arc traversal
times over consecutive node pairs), which is what the public validator does.
"""
import sys, json
from pathlib import Path

CAP = {"S": 10, "M": 20, "L": 30}

def parse_instance(path):
    txt = Path(path).read_text(encoding="utf-8-sig").splitlines()
    n, m, T, c, depot, alpha = txt[0].split()
    n, m, c, depot, T, alpha = int(n), int(m), int(c), int(depot), int(T), float(alpha)
    # detect optional coordinate block
    offset = 1
    if len(txt) >= 1 + n + m + 1 and all(len(txt[i].split()) == 2 for i in range(1, 1 + n)):
        offset += n
    streets = []
    for i in range(m):
        a, b, d, tt, L, cat, req = txt[offset + i].split()
        streets.append(dict(id=i, a=int(a), b=int(b), dir=int(d), time=int(tt),
                            length=int(L), cat=cat, req=int(req)))
    vehicles = txt[offset + m].split()
    # arc -> (street_id, time); respect direction
    arc = {}
    for s in streets:
        arc[(s["a"], s["b"])] = (s["id"], s["time"])
        if s["dir"] == 2:
            arc[(s["b"], s["a"])] = (s["id"], s["time"])
    return dict(n=n, m=m, T=T, depot=depot, alpha=alpha, streets=streets,
                vehicles=vehicles, arc=arc)

def validate(inst, sub_path):
    lines = Path(sub_path).read_text(encoding="utf-8-sig").splitlines()
    errors = []
    streets = inst["streets"]
    C = len(inst["vehicles"])
    if not lines or lines[0].strip() != str(C):
        return False, [f"first line != {C}"], None
    if len(lines) != 1 + 3 * C:
        return False, [f"line count {len(lines)} != {1+3*C}"], None
    cleaned_once = {}      # street_id -> True (coverage, counted once)
    waste = 0.0
    cleaned_global = set()
    for vi in range(C):
        cap = CAP[inst["vehicles"][vi]]
        base = 1 + 3 * vi
        n = int(lines[base].strip())
        nodes = [int(x) for x in lines[base + 1].split()]
        cl = lines[base + 2].split()
        cleaned = [int(x) for x in cl] if cl else []
        if len(nodes) != n + 1:
            errors.append(f"v{vi}: node count {len(nodes)} != n+1={n+1}")
        if not nodes or nodes[0] != inst["depot"] or nodes[-1] != inst["depot"]:
            errors.append(f"v{vi}: not depot-bounded")
            continue
        traversed = {}
        rtime = 0
        ok = True
        for u, v in zip(nodes, nodes[1:]):
            if (u, v) not in inst["arc"]:
                errors.append(f"v{vi}: invalid arc {u}->{v}")
                ok = False
                break
            sid, tt = inst["arc"][(u, v)]
            traversed[sid] = traversed.get(sid, 0) + 1
            rtime += tt
        if not ok:
            continue
        if rtime > inst["T"]:
            errors.append(f"v{vi}: time {rtime} > {inst['T']}")
        for sid in cleaned:
            if sid < 0 or sid >= len(streets):
                errors.append(f"v{vi}: unknown street {sid}"); continue
            s = streets[sid]
            if s["cat"] == "C":
                errors.append(f"v{vi}: connector {sid} cleaned")
            if cap < s["req"]:
                errors.append(f"v{vi}: cap {cap} < req {s['req']} street {sid}")
            if sid not in traversed:
                errors.append(f"v{vi}: cleaned untraversed {sid}")
            cleaned_once[sid] = True
            waste += (cap - s["req"]) * s["length"] / 1000.0
            cleaned_global.add((vi, sid))
    # mandatory coverage
    mand = {s["id"] for s in streets if s["cat"] == "M"}
    missing = mand - set(cleaned_once)
    if missing:
        errors.append(f"missing mandatory: {sorted(missing)[:8]} (+{max(0,len(missing)-8)})")
    # score
    cleanable = [s for s in streets if s["cat"] != "C"]
    Lmax = sum(s["length"] for s in cleanable)
    Wmax = sum((30 - s["req"]) * s["length"] / 1000.0 for s in cleanable)
    cov_len = sum(streets[sid]["length"] for sid in cleaned_once)
    coverage = cov_len / Lmax if Lmax else 1.0
    efficiency = 1.0 if Wmax == 0 else 1.0 - waste / Wmax
    score = inst["alpha"] * coverage + (1 - inst["alpha"]) * efficiency
    valid = not errors
    return valid, errors, dict(score=score, coverage=coverage, efficiency=efficiency,
                               waste=waste, cleaned=len(cleaned_once),
                               mandatory_cleaned=len(mand & set(cleaned_once)),
                               mandatory_total=len(mand))

if __name__ == "__main__":
    inst = parse_instance(sys.argv[1])
    valid, errors, sc = validate(inst, sys.argv[2])
    out = dict(valid=valid, errors=errors[:6])
    if sc: out.update(sc)
    print(json.dumps(out, indent=2))
