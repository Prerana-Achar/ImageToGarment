#!/usr/bin/env python3
"""Explain rejected folders and requirements for larger balanced quotas."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = (
    "top",
    "hoodie",
    "dress",
    "jumpsuit",
    "bottoms",
    "skirts",
    "top_and_bottom",
    "top_and_skirt",
    "hoodie_and_bottom",
    "hoodie_and_skirt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-candidate-folders", type=int, default=20)
    return parser.parse_args()


def inspect_rejected(rejected: list[dict[str, str]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    artifact_counts: Counter[str] = Counter()
    reason_categories: dict[str, Counter[str]] = defaultdict(Counter)
    records = []
    for rejected_record in rejected:
        folder = Path(rejected_record["folder"])
        name = folder.name
        reason = rejected_record["reason"]
        reason_counts[reason] += 1
        metadata_path = folder / f"{name}.json"
        metadata = None
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            pass
        category = metadata.get("category") if isinstance(metadata, dict) else None
        body_name = metadata.get("body_name") if isinstance(metadata, dict) else None
        category_key = str(category) if category in CATEGORIES else "<unknown>"
        category_counts[category_key] += 1
        reason_categories[reason][category_key] += 1

        missing_artifacts = []
        if not metadata_path.is_file():
            missing_artifacts.append("metadata_json_missing")
        elif metadata is None:
            missing_artifacts.append("metadata_json_unreadable")
        pkl_path = folder / f"{name}.pkl"
        if not pkl_path.is_file() or pkl_path.stat().st_size == 0:
            missing_artifacts.append("garment_pkl_missing_or_empty")
        missing_pose = False
        for pose_index in (1, 2, 3):
            pose_path = folder / f"{name}_pose{pose_index}.png"
            if not pose_path.is_file() or pose_path.stat().st_size == 0:
                missing_artifacts.append(f"pose{pose_index}_png_missing_or_empty")
                missing_pose = True
        if "missing/corrupt pose" in reason and not missing_pose:
            missing_artifacts.append("pose_png_exists_but_at_least_one_is_unreadable")
        if not missing_artifacts:
            missing_artifacts.append("metadata_or_target_contract_repair")
        artifact_counts.update(missing_artifacts)
        records.append(
            {
                "folder": str(folder),
                "source_split": rejected_record["split"],
                "reason": reason,
                "category": category,
                "body_name": body_name,
                "missing_artifacts": missing_artifacts,
            }
        )
    return {
        "count": len(rejected),
        "reason_counts": dict(reason_counts.most_common()),
        "category_counts": {
            category: category_counts[category]
            for category in (*CATEGORIES, "<unknown>")
            if category_counts[category]
        },
        "missing_artifact_counts": dict(artifact_counts.most_common()),
        "reason_category_counts": {
            reason: dict(counts.most_common())
            for reason, counts in sorted(reason_categories.items())
        },
        "records": sorted(records, key=lambda item: item["folder"]),
    }


def quota_requirements(
    report: dict[str, Any],
    rejection_audit: dict[str, Any],
    rounds: int,
    max_candidates: int,
) -> dict[str, Any]:
    rejected_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rejection_audit["records"]:
        if record["category"] in CATEGORIES:
            rejected_by_category[str(record["category"])].append(record)
    train = report["train"]
    val = report["val"]
    train_quota = int(train["quota"])
    val_quota = int(val["quota"])
    levels = []
    for increment in range(1, rounds + 1):
        target_train = train_quota + increment
        target_val = val_quota + increment
        by_category = {}
        for category in CATEGORIES:
            train_available = int(train["available"][category])
            val_available = int(val["available"][category])
            train_needed = max(0, target_train - train_available)
            val_needed = max(0, target_val - val_available)
            combined_needed = max(
                0,
                target_train + target_val - train_available - val_available,
            )
            candidates = rejected_by_category[category]
            candidate_bodies = {
                str(record["body_name"])
                for record in candidates
                if record["body_name"] is not None
            }
            by_category[category] = {
                "complete_available_train": train_available,
                "complete_available_val": val_available,
                "needed_in_current_body_partition_train": train_needed,
                "needed_in_current_body_partition_val": val_needed,
                "minimum_additional_complete_global": combined_needed,
                "rejected_repair_candidates": len(candidates),
                "rejected_candidate_bodies": len(candidate_bodies),
                "candidate_folders": [
                    {
                        "folder": record["folder"],
                        "body_name": record["body_name"],
                        "missing_artifacts": record["missing_artifacts"],
                    }
                    for record in candidates[:max_candidates]
                ],
            }
        limiting = [
            category
            for category, item in by_category.items()
            if item["minimum_additional_complete_global"] > 0
            or item["needed_in_current_body_partition_train"] > 0
            or item["needed_in_current_body_partition_val"] > 0
        ]
        levels.append(
            {
                "increment": increment,
                "target_train_per_category": target_train,
                "target_val_per_category": target_val,
                "target_selected_total": len(CATEGORIES) * (target_train + target_val),
                "additional_selected_if_reached": len(CATEGORIES) * 2 * increment,
                "limiting_categories": limiting,
                "by_category": by_category,
            }
        )
    return {
        "current_train_per_category": train_quota,
        "current_val_per_category": val_quota,
        "current_selected_total": int(report["selected_records"]),
        "complete_but_discarded_for_balance": int(report["discarded_complete_records"]),
        "levels": levels,
        "body_disjoint_note": (
            "Train and validation must remain body-disjoint. When both splits need "
            "a limiting category, repair or add that category on at least two bodies "
            "that can be assigned to different splits."
        ),
    }


def print_report(
    rejection_audit: dict[str, Any], requirements: dict[str, Any], output_path: Path
) -> None:
    print("\nRejected folder summary:")
    for reason, count in rejection_audit["reason_counts"].items():
        print(f"  {count:4d}  {reason}")
    print("Missing artifact summary:")
    for artifact, count in rejection_audit["missing_artifact_counts"].items():
        print(f"  {count:4d}  {artifact}")
    print("Rejected folders by recoverable category:")
    for category, count in rejection_audit["category_counts"].items():
        print(f"  {category:24s} {count:4d}")

    print(
        f"\nSelected={requirements['current_selected_total']}; "
        f"complete but discarded for strict balance="
        f"{requirements['complete_but_discarded_for_balance']}"
    )
    first = requirements["levels"][0]
    print(
        f"Next balanced level: {first['target_train_per_category']}/category train + "
        f"{first['target_val_per_category']}/category val = "
        f"{first['target_selected_total']} selected garments "
        f"(+{first['additional_selected_if_reached']})."
    )
    print("Limiting categories for the next level:")
    for category in first["limiting_categories"]:
        item = first["by_category"][category]
        print(
            f"  {category:24s} complete train={item['complete_available_train']:3d} "
            f"val={item['complete_available_val']:3d}; "
            f"minimum additional complete={item['minimum_additional_complete_global']:2d}; "
            f"rejected repair candidates={item['rejected_repair_candidates']:3d}"
        )
        for candidate in item["candidate_folders"][:5]:
            artifacts = ", ".join(candidate["missing_artifacts"])
            print(f"      {candidate['folder']} -> {artifacts}")
    print(requirements["body_disjoint_note"])
    print(f"Full requirements report: {output_path}")


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    prepared = Path(args.prepared_dir).expanduser().resolve()
    report_path = prepared / "balance_report.json"
    if not report_path.is_file():
        raise SystemExit(f"Missing {report_path}")
    report = json.loads(report_path.read_text())
    rejection_audit = inspect_rejected(report.get("rejected", []))
    requirements = quota_requirements(
        report, rejection_audit, args.rounds, args.max_candidate_folders
    )
    output = {
        "format": "balanced_garmentcode_smplx/requirements-v1",
        "prepared_dir": str(prepared),
        "rejections": rejection_audit,
        "quota_requirements": requirements,
    }
    output_path = prepared / "requirements_report.json"
    output_path.write_text(json.dumps(output, indent=2))
    print_report(rejection_audit, requirements, output_path)


if __name__ == "__main__":
    main()

