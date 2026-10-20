"""Bounded real CPU export comparison, with all media retained in .runtime."""
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import main
from stereo import StereoRenderer, make_eye_pair
from video_encoder import EncoderProfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_render_buffers import legacy_xreal


class LegacyRenderer:
    def render(self, frame, depth, strength, convergence, layout):
        assert layout == 'XREAL_SBS'
        return legacy_xreal(*make_eye_pair(frame, depth, strength, convergence))


def decoded(path, ffmpeg):
    capture=cv2.VideoCapture(str(path))
    digest=hashlib.sha256(); count=0; size=None
    try:
        while True:
            ok,frame=capture.read()
            if not ok: break
            count+=1; size=list(frame.shape); digest.update(frame.tobytes())
    finally: capture.release()
    audio=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-map','0:a:0','-f','s16le','-acodec','pcm_s16le','-'],capture_output=True,check=True).stdout
    return dict(frames=count,shape=size,decoded_pixel_sha256=digest.hexdigest(),audio_bytes=len(audio),decoded_audio_sha256=hashlib.sha256(audio).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main_run():
    root=Path(__file__).resolve().parents[1]/'.runtime'/('render-export-verification-'+uuid.uuid4().hex)
    root.mkdir(parents=True)
    ffmpeg=main.FFMPEG_PATH
    source=root/'source.mp4'
    subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','testsrc2=size=1280x720:rate=12:duration=1','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1','-c:v','libx264','-preset','ultrafast','-crf','18','-c:a','aac','-shortest',str(source)],check=True)
    cv2.setNumThreads(1)
    args=('-c:v','libx264','-preset','ultrafast','-crf','22','-threads','1')
    profile=EncoderProfile('fast','libx264',args,args,False)
    trials=[]
    for trial in range(3):
        for label,renderer in ((('old',LegacyRenderer),('new',StereoRenderer)) if trial%2==0 else (('new',StereoRenderer),('old',LegacyRenderer))):
            job=main.Job(id=uuid.uuid4().hex,filename=source.name,input_path=source,media_type='video',layout='XREAL_SBS',strength=5,convergence=.5,depth_mode='balanced',input_size=384,resolution='1080p',quality='fast',invert_depth=False,temporal_smoothing=.35,edge_refinement=True,execution_mode='cpu')
            target=root/f'{trial}-{label}.mp4'
            main.JOBS[job.id]=job
            try:
                with patch.object(main,'StereoRenderer',renderer),patch.object(main,'select_encoder_profile',return_value=profile),patch.object(main,'output_path_for',return_value=target):
                    start=time.perf_counter(); result=main.run_video(job,None,True); elapsed=time.perf_counter()-start
                verification=decoded(result,ffmpeg)
                assert verification['frames']==12 and verification['shape']==[1080,3840,3] and verification['audio_bytes']>0
                trials.append(dict(trial=trial,renderer=label,seconds=elapsed,render_seconds=job.timings.snapshot()['stages']['render']['seconds'],verification=verification))
            finally:
                main.JOBS.pop(job.id,None)
    reference=trials[0]['verification']
    assert all(t['verification']==reference for t in trials), 'Encoded or decoded outputs differ'
    old=statistics.median(t['seconds'] for t in trials if t['renderer']=='old')
    new=statistics.median(t['seconds'] for t in trials if t['renderer']=='new')
    report=dict(root=str(root),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),clip='12 frames 1280x720, 12fps, sine audio',output='XREAL 3840x1080',depth='balanced approximation, CPU',encoder=list(args),opencv_threads=cv2.getNumThreads(),trials=trials,old_median_seconds=old,new_median_seconds=new,speedup=old/new,all_files_and_decoded_frames_and_audio_equal=True,scope='run_video wall time including capture setup, ffprobe, decode, approximation depth, refinement, temporal filtering, rendering, encoding, audio and process completion; excludes model loading, HTTP upload, process_job history/cleanup')
    (root/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main_run()
