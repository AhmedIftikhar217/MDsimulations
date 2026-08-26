#!/usr/bin/env python3
"""
Analysis of the production trajectory.

Three groups of output:

  QUALITY CONTROL -- does the simulation deserve to be believed at all?
    Temperature, density, volume and energy drift; and an a-posteriori
    MINIMUM-IMAGE CHECK.  The box was sized on a PREDICTED worst-case solute
    size (see 00_box_size.py); this measures the size actually reached and
    confirms the prediction held.  If it did not, the trajectory is
    compromised and nothing downstream of it is worth reporting.

  STRUCTURE -- what did individual peptides do?
    Backbone RMSD, radius of gyration per copy, DSSP secondary structure,
    solvent-accessible surface area.

  ASSOCIATION -- what did the peptides do to each other?  This is the actual
    question when you put ten copies in one box.
    Inter-peptide heavy-atom contacts, peptide-peptide contact map, cluster
    sizes over time, and the centre-of-mass radial distribution function.

Periodic boundaries are handled explicitly throughout:
  * contacts use a periodic KD-tree, so a pair that is close only across the
    boundary is still counted;
  * cluster extents are measured on UNWRAPPED coordinates, reconstructed by
    walking the cluster's contact graph -- a cluster split across the boundary
    would otherwise look artificially small, which is exactly the failure the
    minimum-image check exists to catch.

Every quantity is written to CSV and JSON as well as plotted, so numbers go
into the paper without being read off a figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdlib  # noqa: E402
from importlib import import_module  # noqa: E402

boxmod = import_module("00_box_size")

SOLVENT = ("HOH", "WAT", "NA", "CL", "K", "MG", "CA")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_solute(out: Path, stride: int):
    import mdtraj as md
    traj = md.load(str(out / "traj.dcd"), top=str(out / "system.pdb"),
                   stride=stride)
    sel = "not (" + " or ".join(f"resname {r}" for r in SOLVENT) + ")"
    return traj.atom_slice(traj.topology.select(sel))


def peptide_chains(traj) -> list[np.ndarray]:
    """Heavy-atom indices per peptide copy, one array per chain."""
    chains = []
    for chain in traj.topology.chains:
        idx = [a.index for a in chain.atoms if a.element.symbol != "H"]
        if idx:
            chains.append(np.array(idx))
    return chains


def box_edges(traj) -> np.ndarray:
    """Per-frame orthorhombic box edge lengths, shape (n_frames, 3)."""
    return np.array([[v[0][0], v[1][1], v[2][2]] for v in traj.unitcell_vectors])


# ----------------------------------------------------------------------------
# Periodic geometry
# ----------------------------------------------------------------------------

def make_chains_whole(xyz: np.ndarray, chains: list[np.ndarray],
                      edge: np.ndarray) -> list[np.ndarray]:
    """Reassemble each peptide that straddles a periodic boundary.

    Valid because one peptide is far smaller than the box, so every atom is
    within half a box length of the chain's first atom.
    """
    whole = []
    for c in chains:
        ref = xyz[c[0]]
        d = xyz[c] - ref
        d -= edge * np.round(d / edge)
        whole.append(ref + d)
    return whole


def contact_matrix(xyz: np.ndarray, chains: list[np.ndarray], edge: np.ndarray,
                   cutoff: float) -> np.ndarray:
    """Inter-peptide heavy-atom contact counts, under the minimum image.

    A periodic KD-tree is O(N log N) and counts pairs that are neighbours only
    across the boundary -- which a naive wrapped-coordinate distance misses.
    """
    from scipy.spatial import cKDTree

    n = len(chains)
    owner = np.empty(len(xyz), dtype=np.int32)
    for i, c in enumerate(chains):
        owner[c] = i

    flat = np.concatenate(chains)
    # cKDTree's boxsize requires every coordinate inside [0, L).
    pts = np.mod(xyz[flat], edge)
    tree = cKDTree(pts, boxsize=edge)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    counts = np.zeros((n, n), dtype=np.int32)
    if len(pairs):
        a, b = owner[flat[pairs[:, 0]]], owner[flat[pairs[:, 1]]]
        m = a != b
        if m.any():
            np.add.at(counts, (a[m], b[m]), 1)
            np.add.at(counts, (b[m], a[m]), 1)
    return counts


def clusters_from_adjacency(adj: np.ndarray) -> list[list[int]]:
    """Connected components of the peptide-peptide association graph."""
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    out = []
    for start in range(n):
        if seen[start]:
            continue
        comp, q = [], deque([start])
        seen[start] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in np.nonzero(adj[u])[0]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        out.append(sorted(comp))
    return out


def cluster_extents(whole: list[np.ndarray], comps: list[list[int]],
                    adj: np.ndarray, edge: np.ndarray) -> float:
    """Largest bounding-box diagonal over all clusters, UNWRAPPED.

    A cluster spanning the periodic boundary has members on opposite faces.
    Measuring it in wrapped coordinates understates its size -- the opposite of
    conservative, and precisely the case the minimum-image check must catch.
    So each cluster is unwrapped by walking its contact graph and accumulating
    the minimum-image offset between connected members.
    """
    largest = 0.0
    for comp in comps:
        root = comp[0]
        offset = {root: np.zeros(3)}
        q = deque([root])
        while q:
            u = q.popleft()
            cu = whole[u].mean(axis=0) + offset[u]
            for v in np.nonzero(adj[u])[0]:
                if v in offset:
                    continue
                cv = whole[v].mean(axis=0)
                d = cv - cu
                # shift v into the image that actually touches u
                offset[v] = -edge * np.round(d / edge)
                q.append(v)
        pts = np.concatenate([whole[i] + offset[i] for i in comp])
        largest = max(largest, float(np.linalg.norm(pts.max(0) - pts.min(0))))
    return largest


# ----------------------------------------------------------------------------
# Per-frame pass
# ----------------------------------------------------------------------------

def per_frame_pass(traj, chains: list[np.ndarray], cfg: dict) -> dict:
    cutoff = cfg["analysis"]["contact_cutoff_nm"]
    min_contacts = cfg["analysis"]["min_contacts_for_association"]
    n = len(chains)
    edges = box_edges(traj)

    counts = np.zeros((traj.n_frames, n, n), dtype=np.int32)
    largest_cluster = np.zeros(traj.n_frames, dtype=int)
    n_clusters = np.zeros(traj.n_frames, dtype=int)
    max_extent = np.zeros(traj.n_frames)
    coms = np.zeros((traj.n_frames, n, 3))

    for f in range(traj.n_frames):
        edge = edges[f]
        xyz = traj.xyz[f]
        counts[f] = contact_matrix(xyz, chains, edge, cutoff)
        adj = counts[f] >= min_contacts
        comps = clusters_from_adjacency(adj)
        sizes = np.array([len(c) for c in comps])
        largest_cluster[f] = int(sizes.max())
        n_clusters[f] = int((sizes > 1).sum())

        whole = make_chains_whole(xyz, chains, edge)
        coms[f] = np.array([w.mean(axis=0) for w in whole])
        max_extent[f] = cluster_extents(whole, comps, adj, edge)

    return {"contacts": counts, "largest_cluster": largest_cluster,
            "n_clusters": n_clusters, "max_extent": max_extent,
            "coms": coms, "edges": edges}


# ----------------------------------------------------------------------------
# Quality control
# ----------------------------------------------------------------------------

def quality_control(out: Path, cfg: dict, n_atoms: int) -> dict:
    import pandas as pd

    qc: dict = {}
    log = out / "prod_log.csv"
    if not log.exists():
        return qc
    df = pd.read_csv(log)
    df.columns = [c.strip('#" ') for c in df.columns]

    def col(sub):
        hits = [c for c in df.columns if sub.lower() in c.lower()]
        return df[hits[0]].to_numpy(dtype=float) if hits else None

    t = col("Time (ps)")
    t_ns = t / 1000.0 if t is not None else None
    discard = cfg["analysis"]["equilibration_discard_ns"]
    keep = (t_ns >= discard) if t_ns is not None else slice(None)
    if t_ns is not None and not np.any(keep):
        keep = slice(None)   # run shorter than the discard window

    for name, sub in (("temperature_K", "Temperature"),
                      ("density_kg_m3", "Density"),
                      ("volume_nm3", "Box Volume"),
                      ("total_energy_kJ_mol", "Total Energy")):
        v = col(sub)
        if v is None:
            continue
        # OpenMM's StateDataReporter emits density in g/mL; the rest of this
        # pipeline works in kg/m^3, so convert rather than mislabel it.
        if name == "density_kg_m3":
            v = v * 1000.0
        vk = v[keep]
        qc[name] = {"mean": float(vk.mean()), "sd": float(vk.std()),
                    "min": float(vk.min()), "max": float(vk.max())}

    te = col("Total Energy")
    if te is not None and t_ns is not None and len(t_ns[keep]) > 2:
        slope = float(np.polyfit(t_ns[keep], te[keep], 1)[0])
        qc["total_energy_drift_kJ_per_mol_per_ns"] = slope
        qc["total_energy_drift_per_atom_per_ns"] = slope / n_atoms

    if "volume_nm3" in qc:
        qc["mean_concentration_mM"] = boxmod.concentration_from_volume(
            cfg["system"]["n_copies"], qc["volume_nm3"]["mean"])
    return qc


def minimum_image_verdict(pf: dict, cfg: dict) -> dict:
    """Did the largest cluster stay small enough never to see its own image?"""
    cutoff = cfg["box"]["nonbonded_cutoff_nm"]
    edge_min = pf["edges"].min(axis=1)
    clearance = edge_min - pf["max_extent"]
    ok = bool(np.all(clearance > 2 * cutoff))
    worst = int(np.argmin(clearance))
    return {
        "max_cluster_extent_nm": float(pf["max_extent"].max()),
        "mean_cluster_extent_nm": float(pf["max_extent"].mean()),
        "mean_box_edge_nm": float(pf["edges"].mean()),
        "min_clearance_nm": float(clearance.min()),
        "required_clearance_nm": 2 * cutoff,
        "worst_frame": worst,
        "minimum_image_satisfied": ok,
        "verdict": (
            "OK -- the solute never came within a cutoff of its periodic image."
            if ok else
            "VIOLATED -- a cluster grew large enough to interact with its own "
            "periodic image. Rerun in a larger box (raise box.solvent_buffer_nm "
            "or lower box.peptide_concentration_mM); this trajectory cannot "
            "support conclusions about aggregate structure."),
    }


# ----------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------

def structural(traj, chains: list[np.ndarray]) -> dict:
    import mdtraj as md
    res: dict = {}

    bb = traj.topology.select("backbone")
    if len(bb):
        res["backbone_rmsd_nm"] = md.rmsd(traj, traj, 0, atom_indices=bb).tolist()

    rg = np.array([md.compute_rg(traj.atom_slice(c)) for c in chains])
    res["rg_per_peptide_nm"] = rg.tolist()
    res["rg_mean_nm"] = float(rg.mean())
    res["rg_sd_nm"] = float(rg.std())

    try:
        dssp = md.compute_dssp(traj, simplified=True)
        res["dssp_fraction"] = {
            "helix": float((dssp == "H").mean()),
            "sheet": float((dssp == "E").mean()),
            "coil": float((dssp == "C").mean()),
        }
        res["dssp_sheet_per_frame"] = (dssp == "E").mean(axis=1).tolist()
        res["dssp_helix_per_frame"] = (dssp == "H").mean(axis=1).tolist()
    except Exception as exc:
        res["dssp_error"] = repr(exc)

    try:
        sasa = md.shrake_rupley(traj, mode="residue")
        res["sasa_total_nm2_per_frame"] = sasa.sum(axis=1).tolist()
        res["sasa_mean_nm2"] = float(sasa.sum(axis=1).mean())
    except Exception as exc:
        res["sasa_error"] = repr(exc)

    return res


# ----------------------------------------------------------------------------
# RDF between peptide centres of mass
# ----------------------------------------------------------------------------

def com_rdf(pf: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    rmax = cfg["analysis"]["rdf_max_nm"]
    nbins = cfg["analysis"]["rdf_bins"]
    coms, edges = pf["coms"], pf["edges"]
    n_frames, n = coms.shape[0], coms.shape[1]

    bin_edges = np.linspace(0.0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    vols = np.prod(edges, axis=1)
    iu = np.triu_indices(n, 1)

    for f in range(n_frames):
        d = coms[f][:, None, :] - coms[f][None, :, :]
        d -= edges[f] * np.round(d / edges[f])
        hist += np.histogram(np.linalg.norm(d, axis=-1)[iu], bins=bin_edges)[0]

    shell = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    # Ideal-gas expectation for n(n-1)/2 distinct pairs at number density n/V.
    rho = n / vols.mean()
    ideal = shell * rho * (n - 1) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.nan_to_num(hist / n_frames / ideal)
    return 0.5 * (bin_edges[1:] + bin_edges[:-1]), g


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def make_plots(out: Path, cfg: dict, res: dict, t: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, a, mi = res["structure"], res["association"], res["minimum_image"]
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.5))
    fig.suptitle(
        f"{cfg['metadata']['name']}   |   {cfg['system']['n_copies']} x "
        f"{cfg['system']['sequence']}   |   {cfg['production']['length_ns']} ns @ "
        f"{cfg['conditions']['temperature_K']} K, "
        f"{res['quality_control'].get('mean_concentration_mM', float('nan')):.2f} mM",
        fontsize=12)

    if "backbone_rmsd_nm" in s:
        ax[0, 0].plot(t, np.array(s["backbone_rmsd_nm"]) * 10, lw=0.8)
        ax[0, 0].set(xlabel="time (ns)", ylabel="backbone RMSD (Å)",
                     title="Backbone RMSD from start")

    rg = np.array(s["rg_per_peptide_nm"])
    for i in range(rg.shape[0]):
        ax[0, 1].plot(t, rg[i] * 10, lw=0.5, alpha=0.5)
    ax[0, 1].plot(t, rg.mean(axis=0) * 10, lw=2, color="k", label="mean")
    ax[0, 1].set(xlabel="time (ns)", ylabel="Rg (Å)",
                 title="Radius of gyration per peptide")
    ax[0, 1].legend()

    if "dssp_sheet_per_frame" in s:
        ax[0, 2].plot(t, s["dssp_sheet_per_frame"], lw=0.9, label="β-sheet")
        ax[0, 2].plot(t, s["dssp_helix_per_frame"], lw=0.9, label="helix")
        ax[0, 2].set(xlabel="time (ns)", ylabel="fraction of residues",
                     title="Secondary structure (DSSP)")
        ax[0, 2].legend()

    ax[1, 0].plot(t, a["total_contacts_per_frame"], lw=0.8)
    ax[1, 0].set(xlabel="time (ns)", ylabel="inter-peptide contacts",
                 title=f"Heavy-atom contacts < {a['contact_cutoff_nm']} nm")

    ax[1, 1].plot(t, a["largest_cluster_per_frame"], lw=1.0, label="largest cluster")
    ax[1, 1].plot(t, a["n_clusters_per_frame"], lw=1.0, label="n clusters")
    ax[1, 1].set(xlabel="time (ns)", ylabel="peptides", title="Association")
    ax[1, 1].legend()

    ax[1, 2].plot(t, res["_extent"], lw=0.9, label="largest cluster extent")
    ax[1, 2].axhline(mi["mean_box_edge_nm"] - mi["required_clearance_nm"],
                     ls="--", c="r", lw=1.0,
                     label="minimum-image limit")
    ax[1, 2].set(xlabel="time (ns)", ylabel="nm",
                 title="Minimum-image check")
    ax[1, 2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "analysis.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(a["rdf_r_nm"], a["rdf_g"], lw=1.2)
    ax2.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax2.set(xlabel="r between peptide centres of mass (nm)", ylabel="g(r)",
            title="Peptide–peptide radial distribution function")
    fig2.tight_layout()
    fig2.savefig(out / "rdf.png", dpi=150)
    plt.close(fig2)


# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", default="../runs")
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)

    cfg = mdlib.load_config(a.config)
    out = mdlib.run_dir(cfg, a.output_root, a.replicate)
    mdlib.banner(f"ANALYSE  {cfg['metadata']['name']}  replicate {a.replicate}")

    traj = load_solute(out, a.stride)
    chains = peptide_chains(traj)
    print(f"  frames        {traj.n_frames:,}  (stride {a.stride})")
    print(f"  solute        {traj.n_atoms:,} atoms in {len(chains)} chains")

    dt = cfg["production"]["trajectory_interval_ps"] * a.stride / 1000.0
    t = np.arange(traj.n_frames) * dt

    print("\n  per-frame geometry (contacts, clusters, extents) ...", flush=True)
    pf = per_frame_pass(traj, chains, cfg)

    print("  quality control ...", flush=True)
    qc = quality_control(out, cfg, traj.n_atoms)
    mi = minimum_image_verdict(pf, cfg)
    print(f"\n  MINIMUM IMAGE: {mi['verdict']}")
    print(f"    largest cluster extent  {mi['max_cluster_extent_nm']:.2f} nm")
    print(f"    box edge                {mi['mean_box_edge_nm']:.2f} nm")
    print(f"    clearance               {mi['min_clearance_nm']:.2f} nm "
          f"(need > {mi['required_clearance_nm']:.2f})")
    if "mean_concentration_mM" in qc:
        print(f"    simulated concentration {qc['mean_concentration_mM']:.3f} mM")
    if "temperature_K" in qc:
        print(f"    temperature             {qc['temperature_K']['mean']:.2f} "
              f"+/- {qc['temperature_K']['sd']:.2f} K")
    if "density_kg_m3" in qc:
        print(f"    density                 {qc['density_kg_m3']['mean']:.2f} "
              f"+/- {qc['density_kg_m3']['sd']:.2f} kg/m^3")

    print("\n  structure ...", flush=True)
    structure = structural(traj, chains)
    if "dssp_fraction" in structure:
        d = structure["dssp_fraction"]
        print(f"    DSSP  helix {d['helix']:.1%}   sheet {d['sheet']:.1%}   "
              f"coil {d['coil']:.1%}")
    print(f"    Rg per peptide  {structure['rg_mean_nm']*10:.2f} +/- "
          f"{structure['rg_sd_nm']*10:.2f} Å")

    print("\n  association ...", flush=True)
    r, g = com_rdf(pf, cfg)
    assoc = {
        "contact_cutoff_nm": cfg["analysis"]["contact_cutoff_nm"],
        "min_contacts_for_association":
            cfg["analysis"]["min_contacts_for_association"],
        "total_contacts_per_frame": pf["contacts"].sum(axis=(1, 2)).tolist(),
        "mean_contact_matrix": pf["contacts"].mean(axis=0).tolist(),
        "largest_cluster_per_frame": pf["largest_cluster"].tolist(),
        "n_clusters_per_frame": pf["n_clusters"].tolist(),
        "largest_cluster_mean": float(pf["largest_cluster"].mean()),
        "largest_cluster_max": int(pf["largest_cluster"].max()),
        "largest_cluster_final": int(pf["largest_cluster"][-1]),
        "fraction_frames_with_association": float((pf["n_clusters"] > 0).mean()),
        "rdf_r_nm": r.tolist(),
        "rdf_g": g.tolist(),
    }
    print(f"    largest cluster  mean {assoc['largest_cluster_mean']:.2f}, "
          f"max {assoc['largest_cluster_max']}, "
          f"final {assoc['largest_cluster_final']} of {len(chains)}")
    print(f"    frames showing association  "
          f"{assoc['fraction_frames_with_association']:.1%}")

    res = {"replicate": a.replicate, "times_ns": t.tolist(),
           "quality_control": qc, "minimum_image": mi,
           "structure": structure, "association": assoc,
           "_extent": pf["max_extent"].tolist()}
    (out / "analysis.json").write_text(json.dumps(res, indent=2))

    import pandas as pd
    pd.DataFrame({
        "time_ns": t,
        "backbone_rmsd_nm": structure.get("backbone_rmsd_nm",
                                          [np.nan] * len(t)),
        "rg_peptide_mean_nm": np.array(structure["rg_per_peptide_nm"]).mean(axis=0),
        "inter_peptide_contacts": assoc["total_contacts_per_frame"],
        "largest_cluster": assoc["largest_cluster_per_frame"],
        "n_clusters": assoc["n_clusters_per_frame"],
        "largest_cluster_extent_nm": pf["max_extent"],
    }).to_csv(out / "timeseries.csv", index=False)

    if not a.no_plots:
        print("\n  plotting ...", flush=True)
        make_plots(out, cfg, res, t)

    mdlib.write_provenance(cfg, "analysis", out,
                           {"minimum_image": mi, "quality_control": qc})
    print(f"\n  wrote {out}/analysis.json, timeseries.csv, analysis.png, rdf.png")
    if not mi["minimum_image_satisfied"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
