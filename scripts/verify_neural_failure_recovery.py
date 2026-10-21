"""Exercise a real model inference failure followed by queued CPU inference in the frozen EXE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import zlib

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from verify_packaged_release import (MODEL, MODEL_REVISION, MODEL_SHA256, Verifier,
                                     digest, require, utc)


ROOT = Path(__file__).resolve().parents[1]


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def generated_png(path: Path) -> None:
    width, height = 64, 48
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(((x * 4 + y) % 256, (y * 5 + 20) % 256,
                         (x * 3 + y * 2 + 40) % 256))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
                     + png_chunk(b"IDAT", zlib.compress(bytes(rows), 6))
                     + png_chunk(b"IEND", b""))


def malformed_custom_model(path: Path) -> None:
    # This valid graph parses and accepts StereoLab's three-channel input, but
    # its Gather node references a nonexistent input channel at inference time.
    input_info = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 256, 256])
    output_info = helper.make_tensor_value_info("depth", TensorProto.FLOAT, [1, 1, 256, 256])
    indices = numpy_helper.from_array(np.array([5], dtype=np.int64), "indices")
    node = helper.make_node("Gather", ["image", "indices"], ["depth"], axis=1)
    graph = helper.make_graph([node], "queued-inference-failure-fixture", [input_info],
                              [output_info], [indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def create_job(verifier: Verifier, image_path: Path, name: str, mode: str) -> dict:
    settings = {
        "depth_mode": mode,
        "execution_mode": "cpu",
        "layout": "SBS",
        "input_size": "256",
        "resolution": "source",
        "quality": "fast",
        "temporal_smoothing": "0",
        "use_depth_cache": "false",
        "output_name_template": "{name}_{layout}_{model}",
    }
    with image_path.open("rb") as stream:
        admitted = verifier.request("POST", "/api/jobs", data=settings,
                                    files={"file": (name + ".png", stream, "image/png")}).json()
    verifier.report.setdefault("admissions", []).append(admitted)
    verifier.save()
    return admitted


def run(args: argparse.Namespace, report: dict) -> None:
    require(os.name == "nt", "This verifier launches a Windows packaged application.")
    require(args.exe.is_file() and args.ffmpeg.is_file(), "The packaged EXE and configured FFmpeg must exist.")
    require(not args.work.exists(), "--work must name a fresh private directory; existing data is never reused.")
    args.work.mkdir(parents=True)
    args.data = args.work / "isolated-localappdata"
    args.temp = args.work / "private-temp"
    args.report = args.work / "report.json"
    args.data.mkdir()
    args.temp.mkdir()

    # Reuse only the already accepted named model bytes and its pinned receipt.
    source_model = args.models / MODEL / "model.onnx"
    source_receipt = source_model.with_suffix(".accepted.json")
    require(digest(source_model) == MODEL_SHA256, "Source Small model bytes differ from the pinned SHA-256.")
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    require(receipt.get("sha256") == MODEL_SHA256 and receipt.get("revision") == MODEL_REVISION,
            "Source Small model receipt differs from the accepted pinned model.")
    small_dir = args.data / "StereoLab" / ".runtime" / "models" / MODEL
    small_dir.mkdir(parents=True)
    for source in (source_model, source_receipt):
        target = small_dir / source.name
        target.write_bytes(source.read_bytes())
        require(digest(target) == digest(source), f"Model copy changed bytes: {source.name}")
    report["small_model"] = {"sha256": digest(small_dir / "model.onnx"),
                              "receipt": receipt, "copied_without_download": True}

    model_path = args.work / "out-of-range-gather-custom-model.onnx"
    malformed_custom_model(model_path)
    fixture = onnx.load(model_path)
    onnx.checker.check_model(fixture)
    # Confirm the graph loads but fails only on inference, before using the
    # identical fixture in the actual packaged production pipeline.
    probe = cv2.dnn.readNetFromONNX(str(model_path))
    require(bool(probe.getLayerNames()), "OpenCV did not load the custom-model failure fixture.")
    probe.setInput(np.zeros((1, 3, 256, 256), dtype=np.float32))
    try:
        probe.forward()
    except cv2.error as exc:
        probe_failure = str(exc)
        require("outside of range" in probe_failure or "out of range" in probe_failure,
                f"OpenCV failed the fixture for an unexpected reason: {probe_failure}")
    else:
        raise AssertionError("OpenCV accepted the intentionally out-of-range Gather at inference time.")
    report["failure_fixture"] = {
        "path": str(model_path), "sha256": digest(model_path), "bytes": model_path.stat().st_size,
        "input_shape": [1, 3, 256, 256], "invalid_gather_index": 5,
        "preflight": "ONNX checker and OpenCV loading passed; isolated OpenCV inference rejected the out-of-range channel index.",
        "preflight_error": probe_failure,
    }

    image_path = args.work / "generated-64x48.png"
    generated_png(image_path)
    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    require(image is not None and image.shape[:2] == (48, 64), "Generated PNG could not be decoded.")
    report["input_image"] = {"path": str(image_path), "sha256": digest(image_path),
                              "bytes": image_path.stat().st_size, "decoded_shape": list(image.shape)}

    verifier = Verifier(args, report)
    try:
        status = verifier.start()
        server_pid = verifier.request("GET", "/api/diagnostics/resources").json()["pid"]
        report["server_identity"] = {"pid": server_pid,
                                     "path": verifier.report["launches"][-1]["server"]["path"],
                                     "build_fingerprint": status["build_fingerprint"]}
        require(status["build_fingerprint"] == args.expected_fingerprint,
                "The launched EXE does not match the expected source fingerprint.")
        require(verifier.request("GET", "/api/jobs").json() == [], "Fresh packaged API contains unexpected jobs.")

        paused = verifier.request("POST", "/api/queue/control", json={"paused": True}).json()
        require(paused.get("paused") is True, f"Could not pause the isolated queue: {paused}")

        model_response = verifier.request("POST", "/api/model",
            files={"model": (model_path.name, model_path.read_bytes(), "application/octet-stream")}).json()
        require(model_response.get("ok") is True and model_response.get("model_available") is True,
                f"Custom model admission did not pass parsing: {model_response}")
        report["custom_model_admission"] = model_response

        first = create_job(verifier, image_path, "inference-failure", "precision")
        second = create_job(verifier, image_path, "queued-small-recovery", MODEL)
        initial = verifier.request("GET", "/api/jobs").json()
        positions = {job["id"]: job["queue_position"] for job in initial}
        require(positions.get(first["id"]) == 1 and positions.get(second["id"]) == 2,
                f"Failure and recovery were not admitted in queue order: {positions}")
        report["paused_queue"] = {"paused": paused, "positions": positions,
                                  "first_id": first["id"], "second_id": second["id"]}
        verifier.request("POST", "/api/queue/control", json={"paused": False})

        deadline = time.monotonic() + args.job_timeout
        seen_order = False
        transitions = []
        previous = {}
        jobs_by_id = {}
        while time.monotonic() < deadline:
            rows = verifier.request("GET", "/api/jobs").json()
            jobs_by_id = {job["id"]: job for job in rows}
            a, b = jobs_by_id[first["id"]], jobs_by_id[second["id"]]
            for job in (a, b):
                if previous.get(job["id"]) != job["state"]:
                    transition = {"at": utc(), "id": job["id"], "state": job["state"],
                                  "queue_position": job.get("queue_position")}
                    transitions.append(transition)
                    previous[job["id"]] = job["state"]
            if a["state"] in {"failed", "canceled", "completed"} and b["state"] in {"processing", "completed"}:
                seen_order = True
            if a["state"] in {"failed", "canceled", "completed"} and b["state"] in {"failed", "canceled", "completed"}:
                break
            time.sleep(0.025)
        else:
            raise AssertionError("Failure/recovery jobs exceeded the configured timeout.")

        failed_job = jobs_by_id[first["id"]]
        recovery_job = jobs_by_id[second["id"]]
        require(failed_job["state"] == "failed", f"Expected the custom model job to fail: {failed_job}")
        require("Depth inference failed" in (failed_job.get("error") or ""),
                f"The custom model failed outside the expected inference path: {failed_job.get('error')}")
        require(recovery_job["state"] == "completed", f"Queued named-model successor did not complete: {recovery_job}")
        execution = recovery_job.get("execution", {})
        require(execution.get("model_id") == MODEL
                and execution.get("active_provider") == "CPUExecutionProvider"
                and execution.get("inference_succeeded") is True
                and execution.get("gpu_verified") is False
                and not execution.get("fallback_reason"),
                f"Named Small CPU inference provenance is not the expected successful run: {execution}")

        output = verifier.request("GET", f"/api/jobs/{second['id']}/download").content
        decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)
        require(len(output) > 0 and decoded is not None and decoded.shape[0] == 48
                and decoded.shape[1] > 64,
                f"Recovered output was empty or not a decodable side-by-side PNG: {decoded.shape if decoded is not None else None}")
        queue_state = verifier.request("GET", "/api/queue/control").json()
        require(queue_state.get("pending_count") == 0 and queue_state.get("paused") is False,
                f"Queue retained work after recovery: {queue_state}")
        after_pid = verifier.request("GET", "/api/diagnostics/resources").json()["pid"]
        require(after_pid == server_pid, "The application server exited or restarted during failure recovery.")
        require(seen_order, "Polling did not observe the first job fail before its successor began/completed.")

        report["job_transitions"] = transitions
        report["failed_job"] = failed_job
        report["recovery_job"] = recovery_job
        report["recovery_output"] = {
            "sha256": hashlib.sha256(output).hexdigest(), "bytes": len(output),
            "decoded_shape": list(decoded.shape), "decode_succeeded": True,
        }
        report["final_queue"] = queue_state
        report["server_pid_unchanged"] = {"before": server_pid, "after": after_pid}
        report["checks"] = {
            "custom_model_parsed_and_failed_during_real_inference": True,
            "queued_named_small_cpu_inference_completed": True,
            "failure_preceded_successor": True,
            "recovery_output_decoded": True,
            "queue_empty_and_server_alive": True,
        }
        report["passed"] = True
    finally:
        verifier.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / ".runtime/models")
    parser.add_argument("--work", type=Path, required=True, help="Fresh isolated work directory; never reused.")
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--job-timeout", type=float, default=300)
    args = parser.parse_args()
    for name in ("exe", "ffmpeg", "models", "work"):
        setattr(args, name, getattr(args, name).resolve())
    report = {"schema_version": 1, "started_at": utc(), "passed": False,
              "launches": [], "checks": {},
              "scope": "Actual frozen EXE; custom out-of-range ONNX inference failure; queued accepted Small CPU inference",
              "limitations": [
                  "Synthetic generated PNG and deliberately malformed custom-model Gather index.",
                  "This proves runtime inference failure and queue recovery, not naturally occurring model corruption.",
                  "One accepted named model and CPU provider only; no GPU or representative-video claim.",
              ]}
    try:
        run(args, report)
    except Exception as exc:
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
    finally:
        if args.work.exists():
            report["finished_at"] = utc()
            (args.work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(args.work / "report.json"),
                      "checks": report.get("checks"), "failure": report.get("failure")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
