#!/usr/bin/env python3
"""Command-line entrypoint for the six-track research pipeline archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from six_track_pipeline.contracts import STAGES, TRACKS, PipelineContractError
    from six_track_pipeline.loader import default_project_root
    from six_track_pipeline.runner import (
        VerificationFailed,
        build_plan,
        parse_parameters,
        preflight_project,
        verify_pipeline,
    )
else:
    from .contracts import STAGES, TRACKS, PipelineContractError
    from .loader import default_project_root
    from .runner import (
        VerificationFailed,
        build_plan,
        parse_parameters,
        preflight_project,
        verify_pipeline,
    )


def _render(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _failure_payload(exc: Exception) -> dict[str, object]:
    errors = list(exc.errors) if isinstance(exc, PipelineContractError) else [str(exc)]
    return {"schema_version": "six_track_pipeline_error/v1", "status": "FAIL", "errors": errors}


def _write_failure(path: Path, root: Path, payload: dict[str, object]) -> None:
    destination = path if path.is_absolute() else root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root())
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list the six validated adapters")

    plan = subparsers.add_parser("plan", help="show the dependency-closed stage prefix")
    plan.add_argument("--track", choices=(*TRACKS, "all"), required=True)
    plan.add_argument("--through", "--through-stage", dest="through_stage", choices=STAGES, default="verify")
    plan.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")

    preflight = subparsers.add_parser("preflight", help="validate all adapters before any action")
    preflight.add_argument("--intent", choices=("verify", "execute"), default="verify")
    preflight.add_argument("--track", choices=(*TRACKS, "all"), default="all")
    preflight.add_argument("--through", "--through-stage", dest="through_stage", choices=STAGES, default="verify")
    preflight.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")

    verify = subparsers.add_parser("verify", help="verify every lifecycle stage in order")
    verify.add_argument("--track", choices=(*TRACKS, "all"), required=True)
    verify.add_argument("--through", "--through-stage", dest="through_stage", choices=STAGES, default="verify")
    verify.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    try:
        if args.command == "list":
            adapters, preflight = preflight_project(root, intent="verify")
            _render(
                {
                    "schema_version": "six_track_pipeline_list/v1",
                    "preflight": preflight,
                    "pipelines": [
                        {
                            "track": track,
                            "task_dir": adapters[track].task_dir,
                            "manifest": adapters[track].manifest,
                        }
                        for track in TRACKS
                    ],
                }
            )
        elif args.command == "plan":
            params = parse_parameters(args.param)
            _render(
                build_plan(
                    args.track,
                    through_stage=args.through_stage,
                    project_root=root,
                    parameters=params,
                )
            )
        elif args.command == "preflight":
            params = parse_parameters(args.param)
            _, report = preflight_project(
                root,
                intent=args.intent,
                parameters=params,
                track=args.track,
                through_stage=args.through_stage,
            )
            _render(report)
        elif args.command == "verify":
            _render(
                verify_pipeline(
                    args.track,
                    through_stage=args.through_stage,
                    project_root=root,
                    output=args.output,
                )
            )
        else:  # pragma: no cover - argparse makes this unreachable
            parser.error(f"unsupported command: {args.command}")
    except VerificationFailed as exc:
        _render(exc.trace)
        return 1
    except PipelineContractError as exc:
        payload = _failure_payload(exc)
        if args.command == "verify" and args.output is not None:
            _write_failure(args.output, root, payload)
        _render(payload)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
