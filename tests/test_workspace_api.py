import os
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from job_queue import EditableJobQueue


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'workspace-api-tests' / uuid.uuid4().hex
        self.roots = {name: self.root / name for name in ('uploads','previews','outputs','models')}
        self.destination = self.root / 'copies'
        for directory in (*self.roots.values(), self.destination):
            directory.mkdir(parents=True)
        self.addCleanup(self.quarantine)
        for name, value in [('UPLOADS', self.roots['uploads']), ('CLIP_PROXIES', self.roots['previews']),
                            ('OUTPUTS', self.roots['outputs']), ('MODELS', self.roots['models']),
                            ('JOBS', {}), ('JOB_QUEUE', EditableJobQueue()), ('STAGED_UPLOADS', set()), ('HISTORY', None)]:
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)

    def quarantine(self):
        target = main.RUNTIME / '_quarantine' / 'workspace-api-tests' / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, target)

    def job(self, state='completed'):
        identity = str(uuid.uuid4())
        source = self.roots['uploads'] / f'{identity}.png'
        source.write_bytes(b'input')
        job = main.Job(id=identity,filename='fixture.png',input_path=source,media_type='image',
                       layout='SBS',strength=5,convergence=.5,depth_mode='balanced',input_size=256,
                       resolution='source',quality='fast',invert_depth=False,state=state)
        main.JOBS[identity]=job
        if state == 'completed':
            job.output_path=main.output_path_for(job)
            job.output_path.write_bytes(b'completed-output')
        return job

    def test_inventory_and_cleanup_recheck_protected_jobs_and_staged_uploads(self):
        job=self.job('queued')
        staged=self.roots['uploads']/'staged.png';staged.write_bytes(b'upload-in-progress')
        orphan=self.roots['uploads']/'orphan.png';orphan.write_bytes(b'abandoned')
        model=self.roots['models']/'model.onnx';model.write_bytes(b'weights')
        main.STAGED_UPLOADS.add(staged)
        report=self.client.get('/api/storage').json()
        self.assertEqual([c['path'] for c in report['candidates']], [str(orphan)])
        response=self.client.post('/api/storage/cleanup',json={'paths':[str(p) for p in [job.input_path,staged,orphan,model]]})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['deleted'],[str(orphan)])
        self.assertEqual(len(response.json()['errors']),3)
        self.assertTrue(job.input_path.exists());self.assertTrue(staged.exists());self.assertTrue(model.exists())

    def test_stale_candidate_is_protected_after_job_admission(self):
        job=self.job()
        candidate=self.client.get('/api/storage').json()['candidates'][0]['path']
        job.state='processing'
        result=self.client.post('/api/storage/cleanup',json={'paths':[candidate]}).json()
        self.assertEqual(result['deleted'],[])
        self.assertTrue(job.input_path.exists())

    def test_copy_preserves_original_and_refuses_overwrite_and_working_directories(self):
        job=self.job();url=f'/api/jobs/{job.id}/copy'
        response=self.client.post(url,json={'directory':str(self.destination)})
        self.assertEqual(response.status_code,200,response.text)
        copied=Path(response.json()['path'])
        self.assertEqual(copied.read_bytes(),job.output_path.read_bytes())
        copied.write_bytes(b'never-overwrite')
        self.assertEqual(self.client.post(url,json={'directory':str(self.destination)}).status_code,409)
        self.assertEqual(copied.read_bytes(),b'never-overwrite')
        for directory in (self.roots['uploads'],self.roots['models'],self.roots['previews'],self.destination/'missing'):
            self.assertEqual(self.client.post(url,json={'directory':str(directory)}).status_code,409)
        self.assertEqual(job.output_path.read_bytes(),b'completed-output')

    def test_copy_failure_removes_only_new_partial_copy(self):
        job=self.job()
        def fail(reader,writer,**kwargs):
            writer.write(b'partial')
            raise OSError('disk full')
        with patch.object(main.shutil,'copyfileobj',side_effect=fail):
            response=self.client.post(f'/api/jobs/{job.id}/copy',json={'directory':str(self.destination)})
        self.assertEqual(response.status_code,409)
        self.assertEqual(list(self.destination.iterdir()),[])
        self.assertTrue(job.output_path.is_file())

    def test_native_actions_only_resolve_known_completed_outputs(self):
        job=self.job()
        with patch.object(main,'open_folder') as folder, patch.object(main,'reveal_file') as reveal:
            self.assertEqual(self.client.post('/api/output-folder/open').status_code,200)
            self.assertEqual(self.client.post(f'/api/jobs/{job.id}/reveal').status_code,200)
            folder.assert_called_once_with(self.roots['outputs']);reveal.assert_called_once_with(job.output_path)
            job.output_path.unlink()
            self.assertEqual(self.client.post(f'/api/jobs/{job.id}/reveal').status_code,404)

    def test_retry_cleanup_terminal_job_keeps_original_outcome(self):
        job=self.job();job.cleanup_pending[job.input_path]='upload';job.cleanup_warning='locked'
        response=self.client.post(f'/api/jobs/{job.id}/cleanup')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['state'],'completed')
        self.assertIsNone(response.json()['cleanup_warning'])
        self.assertFalse(job.input_path.exists())
        self.assertTrue(job.output_path.exists())
        job.state='processing'
        self.assertEqual(self.client.post(f'/api/jobs/{job.id}/cleanup').status_code,409)

    def test_streamed_upload_is_protected_until_admission(self):
        class Upload:
            filename='fixture.png'
            def __init__(self):self.reads=0
            async def read(self,n):
                self.reads+=1
                if self.reads==1:
                    self_outer.assertEqual(len(main.STAGED_UPLOADS),1)
                    path=next(iter(main.STAGED_UPLOADS))
                    result=main.cleanup_storage({'paths':[str(path)]})
                    self_outer.assertTrue(result['errors'])
                    return b'fixture'
                return b''
            async def close(self):pass
        self_outer=self
        import asyncio
        with patch.object(main,'ensure_worker'):
            result=asyncio.run(main.create_job(file=Upload(),layout='SBS',strength=5,convergence=.5,
                depth_mode='balanced',input_size=256,resolution='source',quality='fast',invert_depth=False,
                temporal_smoothing=.35,edge_refinement=True,execution_mode='cpu',purpose='export',
                clip_start_seconds=0,clip_duration_seconds=3,output_name_template='{name}_{layout}',
                expected_source_sha256=None,request=None))
        self.assertEqual(result['state'],'queued')
        self.assertFalse(main.STAGED_UPLOADS)

    def test_canceled_upload_unregisters_even_when_partial_cleanup_is_locked(self):
        import asyncio
        class Upload:
            filename='fixture.png'
            async def read(self,n):raise asyncio.CancelledError()
            async def close(self):pass
        original=Path.unlink
        def locked(path,*args,**kwargs):
            if path.parent == self.roots['uploads']:raise PermissionError('locked')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',locked):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(main.create_job(file=Upload(),layout='SBS',strength=5,convergence=.5,
                    depth_mode='balanced',input_size=256,resolution='source',quality='fast',invert_depth=False,
                    temporal_smoothing=.35,edge_refinement=True,execution_mode='cpu',purpose='export',
                    clip_start_seconds=0,clip_duration_seconds=3,output_name_template='{name}_{layout}',
                    expected_source_sha256=None,request=None))
        self.assertFalse(main.STAGED_UPLOADS)
        self.assertEqual(len(main.get_storage()['candidates']),1)

    def test_cancel_pending_while_paused_does_not_need_resume_for_cleanup(self):
        job=self.job('queued');main.JOB_QUEUE.put(job.id);main.JOB_QUEUE.set_paused(True)
        response=self.client.post(f'/api/jobs/{job.id}/cancel')
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['state'],'canceled')
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks,0)
        self.assertEqual(main.JOB_QUEUE.pending_ids(),())
        self.assertFalse(job.input_path.exists())
        self.assertTrue(main.JOB_QUEUE.paused)
