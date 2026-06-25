#!/usr/bin/env python3
"""Inspect a ROOT file and dump a structured JSON summary to stdout.

This script is designed to run **inside the CMSSW environment** (via
PyROOT), invoked as a subprocess by combine_runner.py.  It should never
be imported into the bot's own uv-managed Python.

For each top-level object in the file it reports class and name.  For
any RooWorkspace found it additionally enumerates: variables (with
current value and range), PDFs, functions, datasets (with entry count),
category variables, named sets, and snapshots.
"""

from __future__ import annotations

import json
import sys


def _iter_collection(roo_arg_set):
    """Yield each element from a RooAbsCollection / RooArgSet."""
    it = roo_arg_set.createIterator()
    while True:
        obj = it.Next()
        if not obj:
            break
        yield obj


def _summarise_var(v):
    """Return a dict describing a RooRealVar (or RooCategory)."""
    import ROOT

    info = {"name": v.GetName(), "class": v.ClassName()}
    if isinstance(v, ROOT.RooRealVar):
        info["value"] = v.getVal()
        info["min"] = v.getMin()
        info["max"] = v.getMax()
        if v.isConstant():
            info["constant"] = True
    elif isinstance(v, ROOT.RooCategory):
        info["value"] = v.getCurrentLabel()
        info["index"] = v.getCurrentIndex()
    return info


def _summarise_pdf(p):
    return {"name": p.GetName(), "class": p.ClassName()}


def _summarise_data(d):
    return {
        "name": d.GetName(),
        "class": d.ClassName(),
        "entries": d.numEntries(),
    }


def inspect_workspace(ws):
    """Return a dict summarising a RooWorkspace."""
    import ROOT

    result = {"name": ws.GetName(), "title": ws.GetTitle()}

    # Variables (RooRealVar, RooCategory, etc.)
    all_vars = ws.allVars()
    result["variables"] = [_summarise_var(v) for v in _iter_collection(all_vars)]

    # Category variables
    all_cats = ws.allCats()
    result["categories"] = [_summarise_var(c) for c in _iter_collection(all_cats)]

    # PDFs
    all_pdfs = ws.allPdfs()
    result["pdfs"] = [_summarise_pdf(p) for p in _iter_collection(all_pdfs)]

    # Functions (non-pdf)
    all_funcs = ws.allFunctions()
    result["functions"] = [_summarise_pdf(f) for f in _iter_collection(all_funcs)]

    # Datasets
    all_data = ws.allData()
    result["datasets"] = [_summarise_data(d) for d in all_data]

    # Named sets
    named_sets = {}
    sets_map = ws.sets()
    for item in sets_map:
        name = str(item.first)
        members = [obj.GetName() for obj in _iter_collection(item.second)]
        named_sets[name] = members
    result["named_sets"] = named_sets

    # Snapshots
    snapshots = []
    snap_list = ws.getSnapshots()
    if snap_list:
        it = snap_list.MakeIterator()
        while True:
            obj = it.Next()
            if not obj:
                break
            members = [v.GetName() for v in _iter_collection(obj)]
            snapshots.append({"name": obj.GetName(), "parameters": members})
    result["snapshots"] = snapshots

    return result


def inspect_file(path: str) -> dict:
    """Open a ROOT file and return a structured summary."""
    import ROOT

    ROOT.gROOT.SetBatch(True)

    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        return {"error": f"Cannot open ROOT file: {path}"}

    summary: dict = {"file": path, "top_level_objects": [], "workspaces": []}

    for key in f.GetListOfKeys():
        cls_name = key.GetClassName()
        obj_name = key.GetName()
        entry = {"name": obj_name, "class": cls_name}

        if cls_name == "RooWorkspace":
            ws = f.Get(obj_name)
            if ws:
                ws_summary = inspect_workspace(ws)
                summary["workspaces"].append(ws_summary)
                entry["has_detail"] = True

        summary["top_level_objects"].append(entry)

    f.Close()
    return summary


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.root>", file=sys.stderr)
        return 1

    result = inspect_file(sys.argv[1])
    json.dump(result, sys.stdout, indent=2, default=str)
    print()
    return 1 if "error" in result else 0


if __name__ == "__main__":
    raise SystemExit(main())
