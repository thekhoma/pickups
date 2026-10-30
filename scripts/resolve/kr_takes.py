# KR + R1: how Resolve takes behave, run inside Resolve Studio via the Resolve MCP `run_script` sandbox.
# Creates a NEW scratch project (never touches existing projects). `resolve` is pre-injected.
# Media comes from ~/.pickups/resolve_scratch (A.mov 72f @01:00:00:00, A_EXT.mov 120f same TC, B.mov 72f).

# Replaced with the absolute path of ~/.pickups/resolve_scratch/ at run time (the sandbox has no os.path).
MEDIA = "__PICKUPS_SCRATCH__/"
out = {"steps": []}


def log(name, value):
    out["steps"].append({name: value})


def item_facts(it):
    return {
        "name": it.GetName(),
        "start": it.GetStart(),
        "end": it.GetEnd(),
        "duration": it.GetDuration(),
        "left_offset": it.GetLeftOffset(),
        "right_offset": it.GetRightOffset(),
        "takes": it.GetTakesCount(),
        "selected_take": it.GetSelectedTakeIndex(),
        "uid": it.GetUniqueId(),
    }


pm = resolve.GetProjectManager()  # noqa: F821 (Resolve injects `resolve` into its script console)
name = "Pickups Scratch KR"
n = 1
proj = pm.CreateProject(name)
while proj is None and n < 20:
    n += 1
    proj = pm.CreateProject(f"{name} {n}")
log("project", proj.GetName() if proj else None)

proj.SetSetting("timelineFrameRate", "23.976")
mp = proj.GetMediaPool()
clips = mp.ImportMedia([MEDIA + "A.mov", MEDIA + "A_EXT.mov", MEDIA + "B.mov"])
by_name = {c.GetName(): c for c in clips}
log("imported", sorted(by_name))
a, a_ext, b = by_name["A.mov"], by_name["A_EXT.mov"], by_name["B.mov"]

tl = mp.CreateTimelineFromClips("KR", [a, b])
proj.SetCurrentTimeline(tl)
items = tl.GetItemListInTrack("video", 1)
log("before", [item_facts(i) for i in items])
ia, ib = items[0], items[1]

# T1: take extents. Add the longer clip as a take with explicit extents equal to A's visible range.
ok1 = ia.AddTake(a_ext, 0, 71)
log("addtake_0_71", ok1)
log("after_addtake", [item_facts(i) for i in tl.GetItemListInTrack("video", 1)])
log("take_infos", [str(ia.GetTakeByIndex(k)) for k in range(1, ia.GetTakesCount() + 1)])

# T2: a LONGER take (0..119) and what it does to the next clip.
ok2 = ia.AddTake(a_ext, 0, 119)
log("addtake_0_119", ok2)
cnt = ia.GetTakesCount()
log("takes_now", cnt)
sel = ia.SelectTakeByIndex(cnt)
log("select_longest", sel)
log("after_select_longest", [item_facts(i) for i in tl.GetItemListInTrack("video", 1)])
ia.SelectTakeByIndex(1)
log("after_select_original", [item_facts(i) for i in tl.GetItemListInTrack("video", 1)])

# R1: duplicate timeline keeps takes; finalizing on the copy leaves the original intact.
dup = tl.DuplicateTimeline("KR_TURNOVER")
log("duplicate", dup.GetName() if dup else None)
if dup:
    proj.SetCurrentTimeline(dup)
    da = dup.GetItemListInTrack("video", 1)[0]
    log("dup_takes", da.GetTakesCount())
    if da.GetTakesCount() > 1:
        da.SelectTakeByIndex(2)
        log("dup_finalize", da.FinalizeTake())
        log("dup_after_finalize", [item_facts(i) for i in dup.GetItemListInTrack("video", 1)])
    proj.SetCurrentTimeline(tl)
    log("original_after_dup_finalize", [item_facts(i) for i in tl.GetItemListInTrack("video", 1)])

result = out
