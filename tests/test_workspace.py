"""版本保存、任务互斥和刷新恢复的回归检查；只操作临时工作区。"""
import asyncio
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from services import pipeline, storage
from services.job_manager import BusyError, JobManager
from app import routes
from app.main import app


async def request(method, path, body=b"", content_type="application/json", raw=False):
    messages = []
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    route, _, query = path.partition("?")
    await app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
               "http_version": "1.1", "method": method, "scheme": "http",
               "path": route, "raw_path": route.encode(), "query_string": query.encode(),
               "headers": [(b"content-type", content_type.encode())],
               "server": ("testserver", 80), "client": ("127.0.0.1", 1234)}, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    payload = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, payload if raw else json.loads(payload)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.root = Path(self.temp.name)
        data = self.root / "data"
        self.config_patch = patch.multiple(
            config, DATA_ROOT=data, RAW_DIR=data / "raw", SCRAMBLED_DIR=data / "scrambled",
            RESULTS_DIR=self.root / "results", MODEL_DIR=self.root / "models",
            UPLOAD_DIR=data / "uploads", IMAGE_SIZE=16)
        self.config_patch.start()
        self.manager = JobManager(state_file=data / "jobs.json")
        self.manager_patch = patch.object(routes, "job_manager", self.manager)
        self.manager_patch.start()
        self.source = self.root / "source"
        for cls in ("Cat", "Dog"):
            d = self.source / cls
            d.mkdir(parents=True)
            for i in range(5):
                arr = np.random.default_rng(i).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i}.png")

    def tearDown(self):
        self.manager_patch.stop()
        self.config_patch.stop()
        self.temp.cleanup()

    def api(self, method, path, body=None):
        return asyncio.run(request(method, path, json.dumps(body).encode() if body is not None else b""))

    def wait(self, job):
        deadline = time.monotonic() + 10
        while job.to_dict()["status"] in ("pending", "running", "stopping"):
            if time.monotonic() > deadline:
                self.fail("background job timed out")
            time.sleep(.01)
        self.manager.list()  # 等待终态持久化完成后再清理临时目录
        return job.to_dict()

    def import_data(self):
        pipeline.import_dataset(self.source)
        return storage.active_dataset()

    @staticmethod
    def fake_train(**kwargs):
        out = Path(kwargs["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        metrics = {"arch": kwargs["arch"], "best_test_acc": .6}
        storage.write_json(out / "metrics.json", metrics)
        (out / "best.pth").write_bytes(b"test checkpoint")
        kwargs["progress_callback"](50, "half")
        kwargs["progress_callback"](100, "done")
        return metrics

    def test_reservation_blocks_upload_and_all_mutations(self):
        job = self.manager.reserve("import", {})
        for path, body in [("/api/scramble", {}), ("/api/evaluate", {}),
                           ("/api/datasets/select", {"dataset_id": "missing"}),
                           ("/api/original/train-strong", {}),
                           ("/api/original/evaluate-scrambled", {}),
                           ("/api/scrambled/train-adaptive", {})]:
            self.assertEqual(self.api("POST", path, body)[0], 409)
        boundary = "workspace-test"
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="x.zip"\r\n'
                   'Content-Type: application/zip\r\n\r\n').encode() + b"fake" + f"\r\n--{boundary}--\r\n".encode()
        status, _ = asyncio.run(request("POST", "/api/dataset/import", payload,
                                       "multipart/form-data; boundary=" + boundary))
        self.assertEqual(status, 409)
        self.assertFalse(config.UPLOAD_DIR.exists())
        self.manager.finish(job, error="upload cancelled")
        self.assertIsNotNone(self.manager.reserve("scramble", {}))

    def test_concurrent_requests_admit_only_one(self):
        barrier = threading.Barrier(3)
        outcomes = []
        def reserve():
            barrier.wait()
            try:
                self.manager.reserve("evaluate", {})
                outcomes.append("accepted")
            except BusyError:
                outcomes.append("busy")
        threads = [threading.Thread(target=reserve) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.wait()
        for t in threads:
            t.join()
        self.assertCountEqual(outcomes, ["accepted", "busy"])

    def test_failure_releases_lock(self):
        def fail(job):
            raise ValueError("expected failure")
        job = self.manager.submit("evaluate", {}, fail)
        self.assertEqual(self.wait(job)["status"], "error")
        self.assertIsNone(self.manager.active())
        next_job = self.manager.submit("select", {}, lambda j: {"ok": True})
        self.assertEqual(self.wait(next_job)["result"], {"ok": True})

    def test_stop_waits_for_worker_and_releases_lock_with_persisted_status(self):
        started, release = threading.Event(), threading.Event()
        def work(job):
            job.set_progress(35, 'processing')
            started.set()
            if not release.wait(5):
                raise RuntimeError('test timed out')
            job.check_cancelled()
        job = self.manager.submit('evaluate', {}, work)
        self.assertTrue(started.wait(5))
        try:
            self.assertEqual(self.api('POST', f'/api/jobs/{job.id}/stop')[0], 200)
            self.assertEqual(self.api('POST', f'/api/jobs/{job.id}/stop')[0], 200)
            state = self.api('GET', '/api/status')[1]
            self.assertTrue(state['busy'])
            self.assertEqual(state['active_job']['status'], 'stopping')
            self.assertFalse(state['active_job']['can_stop'])
            self.assertEqual(self.api('POST', '/api/evaluate', {})[0], 409)
        finally:
            release.set()
        self.assertEqual(self.wait(job)['status'], 'cancelled')
        self.assertEqual(job.progress, 35)
        self.assertIsNone(self.manager.active())
        recovered = JobManager(state_file=config.DATA_ROOT / 'jobs.json')
        self.assertEqual(recovered.get(job.id).status, 'cancelled')
        self.assertEqual(self.api('POST', f'/api/jobs/{job.id}/stop')[0], 409)
        self.assertEqual(self.api('POST', '/api/jobs/missing/stop')[0], 404)
        self.assertEqual(self.wait(self.manager.submit('select', {}, lambda j: {}))['status'], 'done')

    def test_stop_before_worker_start_and_stop_during_final_commit(self):
        job = self.manager.reserve('import', {})
        self.manager.request_stop(job.id)
        self.manager.start(job, lambda j: self.fail('cancelled job started'))
        self.assertEqual(job.status, 'cancelled')
        finalizing = self.manager.reserve('scramble', {})
        finalizing.begin_commit()
        self.assertFalse(finalizing.to_dict()['can_stop'])
        self.assertEqual(self.api('POST', f'/api/jobs/{finalizing.id}/stop')[0], 409)
        self.manager.finish(finalizing, result={})
        self.assertEqual(finalizing.status, 'done')

    def test_cancel_cleans_all_registered_new_outputs_and_preserves_old_files(self):
        old = config.MODEL_DIR / 'original_classifier' / 'previous'
        old.mkdir(parents=True)
        (old / 'model.pth').write_bytes(b'old model')
        job = self.manager.reserve('evaluate', {})
        with self.assertRaises(ValueError):
            job.track_output(old)
        with self.assertRaises(ValueError):
            job.track_output(self.root / 'unrelated')
        with self.assertRaises(ValueError):
            job.track_output(config.DATA_ROOT / 'datasets')
        folders = [config.DATA_ROOT / 'datasets' / 'new',
                   config.RESULTS_DIR / 'experiments' / 'new',
                   config.MODEL_DIR / 'original_classifier' / 'new',
                   config.MODEL_DIR / 'scrambled_classifier' / 'new']
        for folder in folders:
            job.track_output(folder)
            folder.mkdir(parents=True)
            (folder / 'generated.bin').write_bytes(b'new')
        upload = config.UPLOAD_DIR / 'new.zip'
        job.track_output(upload)
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(b'new upload')
        self.manager.request_stop(job.id)
        self.manager.start(job, lambda j: None)
        self.assertEqual(job.status, 'cancelled')
        self.assertTrue(job.cleanup_complete)
        self.assertTrue(all(not p.exists() for p in folders + [upload]))
        self.assertEqual((old / 'model.pth').read_bytes(), b'old model')

    def test_successful_job_keeps_outputs_and_cleanup_failure_is_reported(self):
        job = self.manager.reserve('scramble', {})
        folder = config.DATA_ROOT / 'datasets' / 'successful'
        job.track_output(folder)
        folder.mkdir(parents=True)
        self.manager.start(job, lambda j: {'ok': True})
        self.assertEqual(self.wait(job)['status'], 'done')
        self.assertTrue(folder.is_dir())
        stopped = self.manager.reserve('scramble', {})
        leftover = config.DATA_ROOT / 'datasets' / 'locked'
        stopped.track_output(leftover)
        leftover.mkdir()
        self.manager.request_stop(stopped.id)
        with patch('services.job_manager.shutil.rmtree', side_effect=PermissionError('file locked')):
            self.manager.start(stopped, lambda j: None)
        self.assertEqual(stopped.status, 'cancelled')
        self.assertFalse(stopped.cleanup_complete)
        self.assertIn('file locked', stopped.cleanup_errors[0])
        self.assertTrue(leftover.exists())
        self.assertIsNone(self.manager.active())

    def test_cleanup_keeps_task_busy_until_deletion_finishes(self):
        started, release, deleting, finish_delete = [threading.Event() for _ in range(4)]
        def work(job):
            started.set()
            release.wait(5)
            job.check_cancelled()
        job = self.manager.submit('evaluate', {}, work)
        self.assertTrue(started.wait(5))
        cleanup = job.cleanup_outputs
        def slow_cleanup():
            deleting.set()
            if not finish_delete.wait(5):
                return ['test timed out']
            return cleanup()
        with patch.object(job, 'cleanup_outputs', side_effect=slow_cleanup):
            try:
                self.manager.request_stop(job.id)
                release.set()
                self.assertTrue(deleting.wait(5))
                self.assertEqual(self.manager.active()['status'], 'stopping')
                self.assertEqual(self.api('POST', '/api/evaluate', {})[0], 409)
            finally:
                finish_delete.set()
            self.assertEqual(self.wait(job)['status'], 'cancelled')
        self.assertTrue(job.cleanup_complete)

    def test_stopped_import_does_not_publish_partial_dataset(self):
        old = self.import_data()
        before = set((config.DATA_ROOT / 'datasets').iterdir())
        job = self.manager.reserve('import', {})
        resize = pipeline.prepare._resize_save
        def stop_after_image(*args):
            result = resize(*args)
            self.manager.request_stop(job.id)
            return result
        with patch.object(pipeline.prepare, '_resize_save', side_effect=stop_after_image):
            self.manager.start(job, lambda j: pipeline.import_dataset(self.source, job=j))
            self.assertEqual(self.wait(job)['status'], 'cancelled')
        self.assertEqual(storage.active_dataset()['id'], old['id'])
        self.assertEqual(len(storage.catalog()['datasets']), 1)
        self.assertTrue(storage.is_ready(old['raw_dir']))
        self.assertEqual(set((config.DATA_ROOT / 'datasets').iterdir()), before)
        self.assertTrue(job.cleanup_complete)

    def test_stopped_scramble_preserves_previous_complete_version(self):
        self.import_data()
        pipeline.run_scramble(iterations=[2])
        old = storage.active_dataset()
        sample = next(Path(old['scrambled_dir']).rglob('*.jpg'))
        before = sample.read_bytes()
        versions = set(Path(old['scrambled_dir']).parent.iterdir())
        job = self.manager.reserve('scramble', {})
        load = pipeline.batch_scramble._load_rgb
        def stop_after_load(*args):
            result = load(*args)
            self.manager.request_stop(job.id)
            return result
        with patch.object(pipeline.batch_scramble, '_load_rgb', side_effect=stop_after_load):
            self.manager.start(job, lambda j: pipeline.run_scramble(iterations=[4], job=j))
            self.assertEqual(self.wait(job)['status'], 'cancelled')
        self.assertEqual(storage.active_dataset()['scramble_id'], old['scramble_id'])
        self.assertEqual(pipeline.get_scrambled_labels(), ['02'])
        self.assertEqual(sample.read_bytes(), before)
        self.assertEqual(set(Path(old['scrambled_dir']).parent.iterdir()), versions)

    def test_stopped_training_removes_new_checkpoint_and_keeps_previous_experiment(self):
        import torch
        self.import_data()
        with patch.object(pipeline.train, 'run', side_effect=self.fake_train):
            previous = pipeline.run_evaluate()
        previous_record = storage.experiment(previous['experiment_id'])
        previous_checkpoint = Path(previous_record['results_dir']) / 'original' / 'best.pth'
        before = previous_checkpoint.read_bytes()
        observed = []
        job = self.manager.reserve('evaluate', {})
        model = torch.nn.Linear(2, 2)
        batches = [(torch.ones(4, 2), torch.tensor([0, 1, 0, 1]))]
        calls = []
        def forward_hook(*args):
            calls.append(1)
            if len(calls) == 2:
                stopped = next(e for e in storage.experiments() if e['id'] != previous['experiment_id'])
                folder = Path(stopped['results_dir']) / 'original'
                self.assertTrue((folder / 'best.pth').exists())
                self.assertEqual(json.loads((folder / 'history.json').read_text())['epoch'], 1)
                observed.append(folder.parent.parent)
                self.manager.request_stop(job.id)
        hook = model.register_forward_hook(forward_hook)
        try:
            with patch.object(pipeline.train, 'build_model', return_value=model), \
                 patch.object(pipeline.train, 'get_dataloaders', return_value=(batches, batches, ['cat', 'dog'])), \
                 patch.object(pipeline.train, 'get_validation_loader', return_value=batches), \
                 patch.object(pipeline.train, 'evaluate', return_value=.8):
                self.manager.start(job, lambda j: pipeline.run_evaluate(arch='resnet18', epochs=3, job=j))
                self.assertEqual(self.wait(job)['status'], 'cancelled')
        finally:
            hook.remove()
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())
        self.assertEqual([e['id'] for e in storage.experiments()], [previous['experiment_id']])
        self.assertEqual(previous_checkpoint.read_bytes(), before)
        self.assertTrue(job.cleanup_complete)
        self.assertEqual(storage.active_dataset()['active_experiment'], previous['experiment_id'])
        self.assertEqual(len(calls), 2)

    def test_stream_upload_can_stop_before_receiving_next_chunk(self):
        self.import_data()
        old = storage.active_dataset()['id']
        async def upload():
            calls = 0
            async def receive():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {'type': 'http.request', 'body': b'partial zip', 'more_body': True}
                active = self.manager.active()
                self.assertEqual(active['kind'], 'import')
                result = await request('POST', '/api/jobs/' + active['id'] + '/stop')
                self.assertEqual(result[0], 200)
                return {'type': 'http.request', 'body': b'rest', 'more_body': False}
            messages = []
            async def send(message):
                messages.append(message)
            await app({'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                       'method': 'POST', 'scheme': 'http', 'path': '/api/dataset/upload',
                       'raw_path': b'/api/dataset/upload', 'query_string': b'filename=test.zip',
                       'headers': [(b'content-type', b'application/zip')],
                       'server': ('testserver', 80), 'client': ('127.0.0.1', 1234)}, receive, send)
            self.assertEqual(next(m['status'] for m in messages if m['type'] == 'http.response.start'), 200)
        asyncio.run(upload())
        self.assertIsNone(self.manager.active())
        self.assertEqual(self.manager.list()[-1]['status'], 'cancelled')
        self.assertEqual(storage.active_dataset()['id'], old)
        self.assertEqual(list(config.UPLOAD_DIR.iterdir()), [])

    def test_stream_upload_completes_normally(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            for path in self.source.rglob('*.png'):
                z.write(path, path.relative_to(self.source))
        code, result = asyncio.run(request('POST', '/api/dataset/upload?filename=cats.zip',
                                           archive.getvalue(), 'application/zip'))
        self.assertEqual(code, 200)
        self.assertEqual(self.wait(self.manager.get(result['job_id']))['status'], 'done')
        self.assertEqual(storage.active_dataset()['name'], 'cats.zip')

    def test_refresh_recovers_progress_and_restart_marks_interruption(self):
        job = self.manager.reserve("import", {"filename": "cats.zip"})
        job.set_progress(42, "loading")
        status, body = self.api("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["active_job"]["progress"], 42)
        recovered = JobManager(state_file=config.DATA_ROOT / "jobs.json")
        self.assertIsNone(recovered.active())
        self.assertEqual(recovered.get(job.id).status, "error")
        self.assertEqual(recovered.get(job.id).params["filename"], "cats.zip")

    def test_reimport_preserves_dataset_and_legacy_results(self):
        from src.data.prepare import import_dataset
        import_dataset(self.source, data_root=config.DATA_ROOT)
        sentinel = config.RESULTS_DIR / "arnold" / "02" / "metrics.json"
        storage.write_json(sentinel, {"arch": "resnet18", "best_test_acc": .75})
        old = storage.active_dataset()
        before = sentinel.read_bytes()
        current = self.import_data()
        self.assertNotEqual(old["id"], current["id"])
        self.assertTrue(Path(old["raw_dir"]).is_dir())
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(len(storage.catalog()["datasets"]), 2)
        self.assertEqual(len(storage.experiments()), 1)
        self.assertEqual(pipeline.get_metrics(old["active_experiment"])["rows"][0]["test_acc"], .75)
        self.assertEqual(pipeline.get_metrics()["rows"], [])
        storage.select_dataset(old["id"])
        self.assertEqual(pipeline.get_metrics()["rows"][0]["test_acc"], .75)

    def test_failed_import_keeps_published_dataset(self):
        old = self.import_data()
        bad = self.root / "invalid.zip"
        bad.write_bytes(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            pipeline.import_dataset(bad)
        self.assertEqual(storage.active_dataset()["id"], old["id"])
        self.assertTrue(Path(old["raw_dir"]).exists())

    def test_partial_import_is_not_advertised_as_ready(self):
        self.import_data()
        job = self.manager.reserve("import", {})
        _, state = self.api("GET", "/api/status")
        self.assertFalse(state["imported"])
        self.assertEqual(state["scrambled"], [])
        self.assertEqual(state["evaluated"], [])
        self.assertEqual(self.api("GET", "/api/metrics")[1]["rows"], [])
        self.manager.finish(job, error="cancelled")
        self.assertTrue(self.api("GET", "/api/status")[1]["imported"])

    def test_repeated_scrambles_do_not_overwrite(self):
        self.import_data()
        pipeline.run_scramble(iterations=[2])
        old = storage.active_dataset()
        sample = next(Path(old["scrambled_dir"]).rglob("*.jpg"))
        contents = sample.read_bytes()
        pipeline.run_scramble(iterations=[4])
        self.assertNotEqual(storage.active_dataset()["scramble_id"], old["scramble_id"])
        self.assertEqual(sample.read_bytes(), contents)
        self.assertEqual(pipeline.get_scrambled_labels(), ["04"])

    def test_repeated_training_keeps_history_and_correct_progress(self):
        self.import_data()
        pipeline.run_scramble(iterations=[2, 4])
        progress = []
        class Reporter:
            def set_progress(self, percent, message):
                progress.append(percent)
        with patch.object(pipeline.train, "run", side_effect=self.fake_train):
            first = pipeline.run_evaluate(arch="resnet18", epochs=3, job=Reporter())
            old = storage.experiment(first["experiment_id"])
            checkpoint = Path(old["results_dir"]) / "02" / "best.pth"
            before = checkpoint.read_bytes()
            second = pipeline.run_evaluate(arch="resnet50", epochs=5)
        self.assertNotEqual(first["experiment_id"], second["experiment_id"])
        self.assertEqual(checkpoint.read_bytes(), before)
        self.assertEqual(len(storage.experiments()), 2)
        self.assertAlmostEqual(progress[0], 100 / 6)
        self.assertEqual(progress[-1], 100)
        self.assertEqual(pipeline.get_metrics()["experiment"]["id"], second["experiment_id"])
        self.assertEqual(pipeline.get_metrics(first["experiment_id"])["rows"][0]["arch"], "resnet18")

    def test_failed_training_preserves_previous_experiment(self):
        self.import_data()
        with patch.object(pipeline.train, "run", side_effect=self.fake_train):
            first = pipeline.run_evaluate()
        with patch.object(pipeline.train, "run", side_effect=RuntimeError("training failed")):
            with self.assertRaises(RuntimeError):
                pipeline.run_evaluate()
        self.assertEqual(pipeline.get_metrics()["experiment"]["id"], first["experiment_id"])
        self.assertEqual(storage.experiments()[0]["status"], "error")

    def test_unknown_history_is_404_and_parameters_are_validated(self):
        self.assertEqual(self.api("GET", "/api/metrics?experiment_id=../../secret")[0], 404)
        self.assertEqual(self.api("POST", "/api/evaluate", {"epochs": 0})[0], 422)
        self.assertEqual(self.api("POST", "/api/scramble", {"iterations": [-1]})[0], 422)

    def saved_model_fixture(self):
        import torch
        from services.frozen_scramble import sha256
        for index, path in enumerate(sorted((self.source / 'Dog').glob('*.png'))):
            Image.fromarray(np.random.default_rng(100 + index).integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
        dataset = self.import_data()
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                    torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
                self.fc = torch.nn.Linear(4, 2)
            def forward(self, x):
                return self.fc(self.features(x).flatten(1))
        torch.manual_seed(7)
        bundle = config.MODEL_DIR / 'original_classifier' / 'source-model'
        bundle.mkdir(parents=True)
        torch.save(Tiny().state_dict(), bundle / 'model.pth')
        storage.write_json(bundle / 'config.json', {'experiment_id': 'source-model', 'arch': 'resnet18',
            'classes': ['cat', 'dog'], 'image_size': 16, 'selection_split': 'val',
            'preprocess': {'resize': [16, 16], 'mean': [0, 0, 0], 'std': [1, 1, 1]}})
        storage.write_json(bundle / 'metrics.json', {'best_test_acc': .5})
        storage.save_experiment({'id': 'source-model', 'dataset_id': dataset['id'], 'dataset_name': dataset['name'],
            'status': 'done', 'created_at': 0, 'parameters': {'arch': 'resnet18'},
            'results_dir': str(storage.experiment_root() / 'source-model' / 'arnold')})
        model_id = 'bundle:original_classifier:source-model'
        return dataset, bundle, model_id, Tiny

    def test_saved_model_api_evaluates_without_training_and_exports_detailed_results(self):
        from services import saved_evaluation
        from services.frozen_scramble import sha256
        dataset, bundle, model_id, Tiny = self.saved_model_fixture()
        original_digest = sha256(bundle / 'model.pth')
        models = self.api('GET', '/api/models')[1]['models']
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]['id'], model_id)
        self.assertNotIn('_path', models[0])
        with patch.object(saved_evaluation, 'build_model', side_effect=lambda *args: Tiny()):
            code, response = self.api('POST', '/api/evaluate/saved', {'model_id': model_id,
                'dataset_id': dataset['id'], 'conditions': 'both', 'iterations': [2], 'batch_size': 2})
            self.assertEqual(code, 200)
            job = self.manager.get(response['job_id'])
            self.assertEqual(self.wait(job)['status'], 'done')
        self.assertEqual(sha256(bundle / 'model.pth'), original_digest)
        result = job.result
        self.assertTrue(result['model_unchanged'])
        self.assertEqual([r['label'] for r in result['summary']], ['original', '02'])
        for row in result['summary']:
            self.assertEqual(row['test_report']['total'], 2)
            self.assertTrue(0 <= row['test_report']['macro_f1'] <= 1)
            self.assertEqual(sum(map(sum, row['test_report']['confusion_matrix'])), 2)
        metrics = self.api('GET', '/api/metrics?experiment_id=' + result['experiment_id'])[1]
        self.assertIsNotNone(metrics['rows'][0]['precision'])
        report_url = f"/api/evaluations/{result['experiment_id']}/download/"
        code, report = self.api('GET', report_url + 'summary')
        self.assertEqual(code, 200)
        self.assertFalse(report['parameters']['weights_updated'])
        self.assertEqual(report['parameters']['data_audit']['checked_source_datasets'], [dataset['id']])
        code, csv_data = asyncio.run(request('GET', report_url + 'predictions', raw=True))
        self.assertEqual(code, 200)
        self.assertEqual(len(csv_data.decode('utf-8-sig').splitlines()), 5)

    def test_saved_evaluation_rejects_missing_inputs_and_training_image_overlap(self):
        from services import saved_evaluation
        dataset, bundle, model_id, Tiny = self.saved_model_fixture()
        body = {'model_id': model_id, 'dataset_id': dataset['id'], 'iterations': [2]}
        self.assertEqual(self.api('POST', '/api/evaluate/saved', {**body, 'model_id': '../../secret'})[0], 422)
        self.assertEqual(self.api('POST', '/api/evaluate/saved', {**body, 'iterations': []})[0], 422)
        self.assertEqual(self.api('POST', '/api/evaluate/saved', {**body, 'iterations': [24]})[0], 422)
        self.assertEqual(self.api('POST', '/api/evaluate/saved', {**body, 'input_source': 'saved'})[0], 422)
        raw = Path(dataset['raw_dir'])
        train = next((raw / 'train' / 'cat').iterdir())
        test = next((raw / 'test' / 'cat').iterdir())
        test.write_bytes(train.read_bytes())
        code, result = self.api('POST', '/api/evaluate/saved', {**body, 'conditions': 'original'})
        self.assertEqual(code, 200)
        stopped = self.wait(self.manager.get(result['job_id']))
        self.assertEqual(stopped['status'], 'error')
        self.assertIn('训练或验证图片', stopped['error'])
        self.assertEqual(len(storage.experiments()), 1)

    def test_saved_evaluation_reads_cached_images_without_scrambling_again(self):
        from services import saved_evaluation
        dataset, bundle, model_id, Tiny = self.saved_model_fixture()
        pipeline.run_scramble(iterations=[2])
        with patch.object(saved_evaluation, 'build_model', side_effect=lambda *args: Tiny()), \
             patch.object(saved_evaluation, 'scramble', side_effect=AssertionError('double scramble')):
            code, response = self.api('POST', '/api/evaluate/saved', {'model_id': model_id,
                'dataset_id': dataset['id'], 'conditions': 'scrambled', 'iterations': [2], 'input_source': 'saved'})
            self.assertEqual(code, 200)
            job = self.manager.get(response['job_id'])
            self.assertEqual(self.wait(job)['status'], 'done')
        self.assertEqual(len(job.result['summary']), 1)
        record = storage.experiment(job.result['experiment_id'])
        self.assertEqual(record['parameters']['input_source'], 'saved')

    def test_stopped_saved_evaluation_cleans_only_new_results_and_preserves_model(self):
        from services import saved_evaluation
        from services.frozen_scramble import sha256
        dataset, bundle, model_id, Tiny = self.saved_model_fixture()
        digest = sha256(bundle / 'model.pth')
        job = self.manager.reserve('evaluate', {})
        model = Tiny()
        hook = model.register_forward_hook(lambda *args: self.manager.request_stop(job.id))
        try:
            with patch.object(saved_evaluation, 'build_model', return_value=model):
                self.manager.start(job, lambda j: saved_evaluation.run(j, model_id, dataset['id'],
                    conditions='both', iterations=[2], batch_size=1))
                self.assertEqual(self.wait(job)['status'], 'cancelled')
        finally:
            hook.remove()
        self.assertTrue(job.cleanup_complete)
        self.assertEqual(sha256(bundle / 'model.pth'), digest)
        self.assertEqual([e['id'] for e in storage.experiments()], ['source-model'])
        self.assertTrue(storage.is_ready(dataset['raw_dir']))

    def test_saved_model_catalog_includes_completed_scratch_checkpoints(self):
        from services import saved_evaluation
        dataset, bundle, model_id, Tiny = self.saved_model_fixture()
        with patch.object(pipeline.train, 'run', side_effect=self.fake_train):
            result = pipeline.run_evaluate(arch='resnet18')
        models = saved_evaluation.public_models()
        self.assertEqual(len(models), 2)
        scratch = next(m for m in models if m['id'].startswith('experiment:'))
        self.assertEqual(scratch['id'], f"experiment:{result['experiment_id']}:original")
        self.assertEqual(scratch['image_size'], 16)

    def test_zip_upload_creates_version_and_keeps_existing_upload(self):
        config.UPLOAD_DIR.mkdir(parents=True)
        old_upload = config.UPLOAD_DIR / "cats.zip"
        old_upload.write_bytes(b"previous upload")
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as z:
            for f in self.source.rglob("*.png"):
                z.write(f, f.relative_to(self.source))
        boundary = "workspace-test"
        payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="cats.zip"\r\n'
                   'Content-Type: application/zip\r\n\r\n').encode()
        payload += archive.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        code, result = asyncio.run(request("POST", "/api/dataset/import", payload,
                                          "multipart/form-data; boundary=" + boundary))
        self.assertEqual(code, 200)
        job = self.manager.get(result["job_id"])
        self.assertEqual(self.wait(job)["status"], "done")
        self.assertEqual(old_upload.read_bytes(), b"previous upload")
        self.assertEqual(storage.active_dataset()["name"], "cats.zip")
        self.assertTrue(self.api("GET", "/api/status")[1]["imported"])

    def test_validation_split_is_preserved_by_scrambling(self):
        pipeline.import_dataset(self.source, validation_ratio=.2)
        dataset = storage.active_dataset()
        self.assertEqual(dataset["counts"], {"train": {"cat": 3, "dog": 3},
                         "val": {"cat": 1, "dog": 1}, "test": {"cat": 1, "dog": 1}})
        pipeline.run_scramble(iterations=[2])
        variant = Path(storage.active_dataset()["scrambled_dir"]) / "arnold_02"
        self.assertEqual(storage.counts(variant), dataset["counts"])

    def test_validation_selects_checkpoint_and_test_is_used_only_once(self):
        import torch
        model = torch.nn.Linear(2, 2)
        train_loader = [(torch.ones(4, 2), torch.tensor([0, 1, 0, 1]))]
        val_loader, test_loader = list(train_loader), list(train_loader)
        calls = []
        hook = model.register_forward_hook(lambda *args: calls.append(1))
        with patch.object(pipeline.train, "build_model", return_value=model), \
             patch.object(pipeline.train, "get_dataloaders", return_value=(train_loader, test_loader, ['cat', 'dog'])), \
             patch.object(pipeline.train, "get_validation_loader", return_value=val_loader), \
             patch.object(pipeline.train, "evaluate", side_effect=[.8, .7, .6]) as evaluate:
            result = pipeline.train.run(self.root, self.root / "training", epochs=2)
        hook.remove()
        self.assertEqual(len(calls), 2)  # 每个训练批次仅前向计算一次
        self.assertIs(evaluate.call_args_list[0].args[1], val_loader)
        self.assertIs(evaluate.call_args_list[1].args[1], val_loader)
        self.assertIs(evaluate.call_args_list[2].args[1], test_loader)
        self.assertEqual(result['best_epoch'], 1)
        self.assertEqual(result['best_test_acc'], .6)
        self.assertTrue(result['test_evaluated_once'])

    def test_exported_linear_head_preserves_standardized_predictions(self):
        import torch
        from services.strong_original import fuse_head
        torch.manual_seed(7)
        features = torch.randn(12, 8)
        mean, scale = torch.randn(8), torch.rand(8) + .1
        head = torch.nn.Linear(8, 2)
        exported = fuse_head(head, mean, scale)
        self.assertTrue(torch.allclose(head((features - mean) / scale), exported(features), atol=1e-5))

    def test_regularized_head_learns_separable_features(self):
        import torch
        from services.strong_original import fit_head, _test_report
        torch.manual_seed(7)
        features = torch.randn(100, 3)
        labels = (features[:, 0] + features[:, 1] > 0).long()
        head, _ = fit_head(features, labels, .001, max_iter=40)
        report = _test_report(head(features), labels, ['cat', 'dog'])
        self.assertGreaterEqual(report['accuracy'], .95)
        self.assertEqual(sum(sum(row) for row in report['confusion_matrix']), 100)
        lower, upper = report['accuracy_95pct_wilson_interval']
        self.assertLessEqual(lower, report['accuracy'])
        self.assertGreaterEqual(upper, report['accuracy'])

    def test_frozen_scramble_preserves_pixels_labels_and_full_period(self):
        import torch
        from torchvision.datasets import ImageFolder
        from services.frozen_scramble import ScrambledTestSet, sha256
        from src.scrambling.arnold import period
        folder = ImageFolder(self.source)
        preprocess = {"resize": [16, 16], "mean": [0, 0, 0], "std": [1, 1, 1]}
        original = ScrambledTestSet(folder, preprocess, 0)
        changed = ScrambledTestSet(folder, preprocess, 2)
        restored = ScrambledTestSet(folder, preprocess, period(16))
        digest = sha256(folder.samples[0][0])
        a, label = original[0]
        b, changed_label = changed[0]
        c, restored_label = restored[0]
        self.assertEqual((label, label), (changed_label, restored_label))
        self.assertTrue(torch.equal(a, c))
        self.assertFalse(torch.equal(a, b))
        for channel in range(3):
            self.assertTrue(torch.equal(a[channel].flatten().sort().values,
                                        b[channel].flatten().sort().values))
        self.assertEqual(digest, sha256(folder.samples[0][0]))

    def test_adaptive_training_updates_backbone_and_head(self):
        import torch
        from services.adaptive_scramble import build_optimizer, train_epoch, evaluate
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                                    torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
                self.fc = torch.nn.Linear(4, 2)
            def forward(self, images):
                return self.fc(self.features(images).flatten(1))
        torch.manual_seed(7)
        model = Tiny().requires_grad_(False)
        optimizer = build_optimizer(model, .001, .01, .01)
        old_conv, old_head = model.features[0].weight.detach().clone(), model.fc.weight.detach().clone()
        batches = [(torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 0, 1]))]
        train_epoch(model, batches, optimizer, lambda *args: None, use_bf16=False)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(torch.equal(old_conv, model.features[0].weight))
        self.assertFalse(torch.equal(old_head, model.fc.weight))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        evaluate(model, batches)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))

    def test_augmentation_precedes_scramble_and_validation_is_fixed(self):
        import torch
        from torchvision import datasets, transforms
        from services.frozen_scramble import ScrambledTestSet, sha256
        from services.adaptive_scramble import build_augmentation
        from src.scrambling.arnold import scramble
        folder = datasets.ImageFolder(self.source)
        preprocess = {"resize": [16, 16], "mean": [0, 0, 0], "std": [1, 1, 1]}
        flip = transforms.RandomHorizontalFlip(p=1.)
        train = ScrambledTestSet(folder, preprocess, 2, pre_scramble=flip)
        validation = ScrambledTestSet(folder, preprocess, 2)
        path = folder.samples[0][0]
        digest = sha256(path)
        with Image.open(path) as image:
            raw = np.asarray(image.convert('RGB'))
        expected = transforms.ToTensor()(scramble(raw[:, ::-1].copy(), 2))
        wrong_order = transforms.ToTensor()(scramble(raw, 2)[:, ::-1].copy())
        tensor, label = train[0]
        self.assertTrue(torch.equal(tensor, expected))
        self.assertFalse(torch.equal(tensor, wrong_order))
        first, val_label = validation[0]
        stochastic = ScrambledTestSet(folder, preprocess, 2, pre_scramble=build_augmentation([16, 16]))
        augmented = [stochastic[0][0] for _ in range(3)]
        self.assertTrue(all(t.shape == (3, 16, 16) for t in augmented))
        self.assertTrue(any(not torch.equal(augmented[0], t) for t in augmented[1:]))
        self.assertTrue(torch.equal(first, validation[0][0]))
        self.assertEqual(label, val_label)
        self.assertEqual(sha256(path), digest)

    def test_default_scrambled_model_uses_validation_instead_of_test_score(self):
        from services.adaptive_scramble import validation_promotion
        bundle = config.MODEL_DIR / 'previous'
        parameters = {"source_model_sha256": 'source', "source_split_manifest_sha256": 'split', "iterations": [2]}
        storage.write_json(bundle / 'metrics.json', {"parameters": parameters,
            "best_validation_acc": .8, "best_validation_loss": .4, "test_accuracy": .99})
        storage.write_json(config.MODEL_DIR / 'scrambled_classifier.json',
                           {"bundle_dir": str(bundle), "experiment_id": 'previous'})
        self.assertEqual(validation_promotion((.79, -.1), parameters), (False, 'previous'))
        self.assertEqual(validation_promotion((.8, -.3), parameters), (True, 'previous'))
        self.assertEqual(validation_promotion((.81, -.9), parameters), (True, 'previous'))

    def test_adaptive_run_selects_validation_and_preserves_original(self):
        import torch
        from services import adaptive_scramble
        from services.frozen_scramble import sha256
        from services.job_manager import Job
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                                    torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1))
                self.fc = torch.nn.Linear(4, 2)
            def forward(self, images):
                return self.fc(self.features(images).flatten(1))
        for index, path in enumerate(sorted((self.source / 'Dog').glob('*.png'))):
            Image.fromarray(np.random.default_rng(200 + index).integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)
        pipeline.import_dataset(self.source, validation_ratio=.2)
        dataset = storage.active_dataset()
        raw = Path(dataset['raw_dir'])
        manifest = [{"path": str(path), "split": path.parent.parent.name, "sha256": sha256(path)}
                    for path in raw.rglob('*') if path.suffix.lower() in ('.jpg', '.png', '.jpeg')]
        storage.write_json(self.root / 'manifest.json', manifest)
        dataset = storage.update_dataset(dataset['id'], split_manifest=str(self.root / 'manifest.json'))
        bundle = config.MODEL_DIR / 'source'
        bundle.mkdir(parents=True)
        torch.save(Tiny().state_dict(), bundle / 'model.pth')
        original_digest = sha256(bundle / 'model.pth')
        settings = {"classes": ['cat', 'dog'], "image_size": 16,
                    "preprocess": {"resize": [16, 16], "mean": [0, 0, 0], "std": [1, 1, 1]}}
        storage.write_json(bundle / 'config.json', settings)
        storage.write_json(config.MODEL_DIR / 'original_classifier.json',
                           {"bundle_dir": str(bundle), "experiment_id": 'source-model'})
        storage.save_experiment({"id": 'source-model', "status": 'done', "dataset_id": dataset['id'],
                                 "created_at": 0, "parameters": {}})
        good = torch.tensor([[2., 0.], [0., 2.]])
        weak = torch.tensor([[2., 0.], [2., 0.]])
        targets = torch.tensor([0, 1])
        calls = []
        def score(model, loader, callback=None):
            calls.append((loader.dataset.folder.root, loader.dataset.pre_scramble))
            return (good if len(calls) == 1 else weak), targets, float(len(calls))
        with patch.object(adaptive_scramble.models, 'resnet18', side_effect=lambda **kwargs: Tiny()), \
             patch.object(adaptive_scramble, 'evaluate', side_effect=score):
            result = adaptive_scramble.run(Job('evaluate', {}), epochs=2, batch_size=2, augmentation=True)
        self.assertEqual([Path(path).name for path, _ in calls], ['val', 'val', 'train', 'test'])
        self.assertTrue(all(augmentation is None for _, augmentation in calls))
        self.assertEqual(result['best_epoch'], 1)
        self.assertEqual(result['test_accuracy'], .5)
        self.assertEqual(sha256(bundle / 'model.pth'), original_digest)
        exported = Path(result['model_bundle'])
        self.assertTrue((exported / 'model.pth').exists())
        self.assertNotEqual(sha256(exported / 'model.pth'), original_digest)
        metrics = json.loads((exported / 'metrics.json').read_text(encoding='utf-8'))
        self.assertTrue(metrics['test_evaluated_once'])
        self.assertTrue(metrics['parameters']['all_layers_trainable'])
        self.assertEqual(metrics['parameters']['augmentation'], 'pre_scramble_light_v1')
        self.assertFalse(metrics['parameters']['augmentation_settings']['validation_and_test_augmented'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
