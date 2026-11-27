"""HZN-EPI-01 — discriminating-query selection probe.

Tests whether the model can identify which additional observation would distinguish
two rules that are consistent with all current evidence. Deterministic construction;
inference-only grading.
"""
import hashlib, random
EPI_VERSION=1
CONDITIONS=("NOCOST","COST")
RULEPAIRS=(("conj","sum"),("conj","either"),("conj","neg"))

def _seed(*parts):
    return int.from_bytes(hashlib.sha256(("epi-"+":".join(map(str,parts))).encode()).digest()[:8],"little")

def make_rule(kind,rng):
    if kind=="conj": return {"kind":kind,"a":round(rng.uniform(.35,.55),2),"b":round(rng.uniform(.25,.45),2)}
    if kind=="sum": return {"kind":kind,"a":round(rng.uniform(.30,.50),2),"c":round(rng.uniform(.75,1.05),2)}
    if kind=="either": return {"kind":kind,"a":round(rng.uniform(.55,.75),2),"b":round(rng.uniform(.55,.75),2)}
    if kind=="neg": return {"kind":kind,"a":round(rng.uniform(.35,.55),2),"b":round(rng.uniform(.25,.55),2)}
    raise ValueError(kind)

def eval_rule(r,x,y):
    if r["kind"]=="conj": return x>=r["a"] and y>=r["b"]
    if r["kind"]=="sum": return x>=r["a"] and x+y>=r["c"]
    if r["kind"]=="either": return x>=r["a"] or y>=r["b"]
    if r["kind"]=="neg": return x>=r["a"] and y<=r["b"]
    raise ValueError(r["kind"])

def rule_text(r):
    if r["kind"]=="conj": return f"pass if x >= {r['a']:.2f} AND y >= {r['b']:.2f}"
    if r["kind"]=="sum": return f"pass if x >= {r['a']:.2f} AND x + y >= {r['c']:.2f}"
    if r["kind"]=="either": return f"pass if x >= {r['a']:.2f} OR y >= {r['b']:.2f}"
    if r["kind"]=="neg": return f"pass if x >= {r['a']:.2f} AND y <= {r['b']:.2f}"

def case(rng): return (round(rng.uniform(0,1),2),round(rng.uniform(0,1),2))

def make_epistemic_task(version,family,pair,rep):
    rng=random.Random(_seed(version,family,pair,rep))
    ra,rb=make_rule(pair[0],rng),make_rule(pair[1],rng)
    if rng.random()<.5: ra,rb=rb,ra
    obs=[]
    if family=="already_resolved":
        winner=rng.choice([0,1]); rules=[ra,rb]
        # decisive observations make only winner consistent.
        while len(obs)<4:
            x,y=case(rng); pa,pb=eval_rule(ra,x,y),eval_rule(rb,x,y)
            if pa==pb: outcome=pa
            elif (pa if winner==0 else pb)!=(pb if winner==0 else pa):
                # choose outcome consistent with only the selected survivor
                outcome=pa if winner==0 else pb
                if eval_rule(rules[winner],x,y)!=outcome or eval_rule(rules[1-winner],x,y)==outcome: continue
            else: continue
            if len(obs)<3 and pa!=pb: continue
            if len(obs)>=3 and pa==pb: continue
            obs.append({"x":x,"y":y,"outcome":outcome})
    else:
        while len(obs)<4:
            x,y=case(rng); pa,pb=eval_rule(ra,x,y),eval_rule(rb,x,y)
            if pa==pb: obs.append({"x":x,"y":y,"outcome":pa})
    for attempt in range(10000):
        qs=[case(rng) for _ in range(5)]
        div=[eval_rule(ra,x,y)!=eval_rule(rb,x,y) for x,y in qs]
        if family=="need_query" and sum(div)==1: break
        if family!="need_query" and sum(div)==0: break
    else: raise AssertionError("query construction failed")
    order=list(range(5)); rng.shuffle(order)
    queries=[]; correct=None
    for label_i,old_i in enumerate(order):
        q={"label":f"Q{label_i+1}","x":qs[old_i][0],"y":qs[old_i][1],"discriminates":div[old_i]}
        queries.append(q)
        if q["discriminates"]: correct=q["label"]
    truth=correct if family=="need_query" else "no_query"
    return {"task_id":f"e{len(pair)}_{family}_{rep}","family":family,"pair":pair,
            "rules":[ra,rb],"observations":obs,"queries":queries,"truth":truth,
            "correct_discriminating_label":correct}

def build_epistemic_tasks(version=EPI_VERSION):
    tasks=[]
    for family in ("need_query","already_resolved","query_irrelevant"):
        for rep in range(8):
            pair=RULEPAIRS[rep%len(RULEPAIRS)]
            t=make_epistemic_task(version,family,pair,rep); t["task_id"]=f"e{len(tasks):02d}"
            tasks.append(t)
    return tasks

def render_epistemic_task(task,condition):
    lines=["Two candidate rules are under consideration.",
           f"Rule 1: {rule_text(task['rules'][0])}.",f"Rule 2: {rule_text(task['rules'][1])}.",
           "","Observed outcomes:"]
    for o in task["observations"]:
        lines.append(f"x={o['x']:.2f}, y={o['y']:.2f} -> {'pass' if o['outcome'] else 'fail'}")
    lines += ["","You may request exactly one additional labeled outcome before deciding which rule applies.",
              "Available query cases:"]
    for q in task["queries"]: lines.append(f"{q['label']}: x={q['x']:.2f}, y={q['y']:.2f}")
    if condition=="COST": lines.append("A query costs 5 resource units. Choosing no_query costs 0.")
    elif condition!="NOCOST": raise ValueError(condition)
    lines += ["","If one query can distinguish the rules, choose that query. If no additional query is needed or none of these queries can distinguish them, choose no_query.",
              "Reply with ONLY choice=<Q1-Q5 or no_query>."]
    return "\n".join(lines)

def parse_choice(response,task):
    text=str(response).strip()
    valid=[q["label"] for q in task["queries"]]+["no_query"]
    if text.startswith("choice="): text=text.split("=",1)[1].strip()
    return text if text in valid else None
