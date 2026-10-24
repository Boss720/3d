"""Guided workflow contracts and actual shipped HTML; no fabricated DOM."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from tests.test_frontend_markup import MarkupTree

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


class GuidedWorkflowTests(unittest.TestCase):
    def contract(self, body):
        if not NODE:
            self.skipTest("Node.js is required")
        script = "const assert=require('node:assert/strict');const c=require(" + json.dumps(str(ROOT / "static/app.js")) + ");" + body
        result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_quick_choices_map_to_existing_settings_and_preserve_device_layout(self):
        self.contract("""
for (const [device, layout] of Object.entries({xreal:'XREAL_SBS',full:'SBS',half:'HALFSBS',anaglyph:'ANAGLYPH'})) {
  for (const quality of ['fast','balanced','quality']) {
    const settings=c.quickConvertSettings(device,quality);
    assert.deepEqual(settings,{...c.performancePreset(quality),layout});
    assert.equal(c.performanceSettingsForLayout(quality,layout).layout,layout);
  }
}
assert.throws(()=>c.quickConvertSettings('unknown','fast'),/device/i);
assert.throws(()=>c.quickConvertSettings('xreal','unknown'),/quality/i);
assert.equal(c.performancePreset('balanced').layout,'XREAL_SBS');
""")

    def test_pins_hold_exact_settings_and_reject_stale_or_different_frames(self):
        self.contract("""
const context={sourceId:'source-1',frameTime:1.125,detail:'proxy'};
const settings=c.performancePreset('balanced');
const result={image:'data:image/png;base64,AA==',depth:'data:image/png;base64,AA==',width:1920,height:540};
const pin=c.createStillPin(result,settings,context,3,3);
settings.strength=9;context.frameTime=7;
assert.equal(pin.settings.strength,5);assert.equal(pin.context.frameTime,1.125);
assert.equal(c.stillContextMatches(pin.context,{sourceId:'source-1',frameTime:1.125,detail:'proxy'}),true);
for(const change of [{sourceId:'source-2'},{frameTime:1.126},{detail:'full'}]) {
 assert.equal(c.stillContextMatches(pin.context,{...pin.context,...change}),false);
}
assert.equal(c.stillContextMatches(null,pin.context),false);
assert.throws(()=>c.createStillPin(result,settings,pin.context,3,4),/fresh|stale/i);
assert.throws(()=>c.createStillPin(null,settings,pin.context,3,3),/preview/i);
assert.throws(()=>c.createStillPin(result,settings,{...pin.context,frameTime:NaN},3,3),/context/i);
""")

    def test_cache_guidance_never_predicts_a_hit(self):
        self.contract("""
assert.match(c.depthReuseGuidance(false,'video'),/off/i);
assert.match(c.depthReuseGuidance(true,'image'),/video.*only/i);
const message=c.depthReuseGuidance(true,'video');
assert.match(message,/verified.*export/i);
assert.match(message,/model|input/i);
assert.doesNotMatch(message,/will reuse|cache hit|guaranteed/i);
""")

    def test_experimental_settings_are_explicit_and_legacy_settings_default_safe(self):
        self.contract("""
const legacy=c.performancePreset('balanced');
const {temporal_depth_scale,renderer,...old}=legacy;
assert.equal(c.validateSettings(old).temporal_depth_scale,false);
assert.equal(c.validateSettings(old).renderer,'fast');
assert.equal(c.validateSettings({...legacy,depth_interval:0}).depth_interval,0);
assert.equal(c.validateSettings({...legacy,renderer:'visibility',temporal_depth_scale:true}).renderer,'visibility');
assert.throws(()=>c.validateSettings({...legacy,renderer:'unknown'}),/renderer/i);
assert.throws(()=>c.validateSettings({...legacy,temporal_depth_scale:'true'}),/boolean/i);
""")

    def test_export_measurement_requires_bounded_validated_outputs_and_complete_evidence(self):
        self.contract("""
const settings=c.performancePreset('balanced');
const candidate=id=>({candidate_id:id,settings,samples_seconds:[2,3],median_seconds:2.5,error:null,
 runs:[0,1].map(repeat_index=>({repeat_index,completed:true,validated:true,result:{completed:true,validated:true,execution:{active_provider:'CPUExecutionProvider'},output:{url:`/api/calibration-exports/job-1/${id}/${repeat_index}`}}}))});
const report={schema_version:1,state:'completed',sufficient_evidence:true,settings_applied:false,identity:{sample:{scope:'bounded_proxy',start_seconds:2,duration_seconds:3}},candidates:[candidate('baseline'),candidate('input256')],recommendation:{candidate_id:'input256',settings,quality_verified:false},scope:'bounded sample',limitations:[]};
assert.equal(c.validateExportComparison(report,'job-1').candidates.length,2);
assert.throws(()=>c.validateExportComparison({...report,settings_applied:true},'job-1'),/report/i);
const unsafe=structuredClone(report);unsafe.candidates[0].runs[0].result.output.url='https://example.com/video.mp4';
assert.throws(()=>c.validateExportComparison(unsafe,'job-1'),/output/i);
const partial=structuredClone(report);partial.candidates[0].runs.pop();
assert.throws(()=>c.validateExportComparison(partial,'job-1'),/evidence/i);
assert.equal(c.exportComparisonContextMatches({sourceId:'s',frameTime:2,settings:'{}',duration:3,budget:30},{sourceId:'s',frameTime:2,settings:'{}',duration:3,budget:60}),false);
""")

    def test_guided_controls_have_real_labels_buttons_and_still_only_copy(self):
        tree = MarkupTree((ROOT / "static/index.html").read_text(encoding="utf-8"))
        nodes = {n["attrs"]["id"]: n for n in tree.elements if "id" in n["attrs"]}
        for element_id in ("quickDevice", "quickQuality"):
            self.assertTrue(element_id in nodes, f"Missing real control {element_id}")
            self.assertEqual(nodes[element_id]["tag"], "select")
            self.assertTrue(any(n["tag"] == "label" and n["attrs"].get("for") == element_id for n in tree.elements))
        for element_id in ("applyQuickConvert", "pinStillA", "pinStillB", "applyStillA", "applyStillB", "clearStillPins"):
            self.assertTrue(element_id in nodes, f"Missing real button {element_id}")
            self.assertEqual(nodes[element_id]["tag"], "button")
            self.assertEqual(nodes[element_id]["attrs"].get("type"), "button")
        for element_id in ("quickConvertStatus", "stillComparisonStatus", "depthReuseStatus"):
            self.assertEqual(nodes[element_id]["attrs"].get("role"), "status")
        for element_id in ("stillImageA", "stillImageB"):
            self.assertEqual(nodes[element_id]["tag"], "img")
            self.assertIn("alt", nodes[element_id]["attrs"])
        html = (ROOT / "static/index.html").read_text(encoding="utf-8")
        self.assertIn("same source and timestamp", html)
        self.assertIn("still frames", html)

    def test_calibration_poll_uses_existing_job_list_api_and_real_sample_players(self):
        script = (ROOT / "static/app.js").read_text(encoding="utf-8")
        poll = script.split("async function pollExportComparison()", 1)[1].split("async function cancelExportComparison()", 1)[0]
        self.assertIn('request("/api/jobs")', poll)
        self.assertNotIn('request(`/api/jobs/${', poll)
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertRegex(main, r'@app\.get\([\"\x27]/api/jobs[\"\x27]\)')
        tree = MarkupTree((ROOT / "static/index.html").read_text(encoding="utf-8"))
        nodes = {n["attrs"]["id"]: n for n in tree.elements if "id" in n["attrs"]}
        for element_id in ("exportBaselineVideo", "exportCandidateVideo"):
            self.assertEqual(nodes[element_id]["tag"], "video")
            self.assertIn("controls", nodes[element_id]["attrs"])
        for element_id in ("startExportComparison", "cancelExportComparison", "applyExportRecommendation"):
            self.assertEqual(nodes[element_id]["tag"], "button")

    def test_recovery_contract_requires_full_source_restart_and_explicit_opt_in(self):
        self.contract("""
const report={capabilities:{enabled:true,retained_jobs:1,retained_bytes:42,max_records:8,max_bytes:100,resume_mid_video:false,restart_mode:'full_source'},items:[{job_id:'job-1',filename:'clip.mp4',state:'interrupted',frames_processed:20,source_size_bytes:42,source_present:true,resume_mid_video:false,restart_mode:'full_source'}]};
assert.equal(c.validateRecoveryReport(report).items.length,1);
assert.equal(c.retainSourceForSubmission({}),false);
assert.equal(c.retainSourceForSubmission({retainSource:'true'}),false);
assert.equal(c.retainSourceForSubmission({retainSource:true}),true);
assert.equal(c.recoveryRestartAllowed(report.items[0],report.capabilities),true);
assert.equal(c.recoveryRestartAllowed({...report.items[0],source_present:false},report.capabilities),false);
assert.equal(c.recoveryRestartAllowed({...report.items[0],state:'processing'},report.capabilities),false);
assert.equal(c.recoveryRestartAllowed({...report.items[0],state:'claimed'},report.capabilities),false);
assert.equal(c.recoveryRestartAllowed(report.items[0],{...report.capabilities,enabled:false}),false);
assert.throws(()=>c.validateRecoveryReport({...report,capabilities:{...report.capabilities,resume_mid_video:true}}),/recovery/i);
assert.throws(()=>c.validateRecoveryReport({...report,items:[{...report.items[0],source_present:'yes'}]}),/recovery/i);
""")

    def test_recovery_has_real_status_and_accurate_forget_copy(self):
        html=(ROOT / "static/index.html").read_text(encoding="utf-8")
        tree=MarkupTree(html)
        nodes={n["attrs"]["id"]:n for n in tree.elements if "id" in n["attrs"]}
        self.assertTrue("recoveryStatus" in nodes,"Missing recovery status")
        self.assertEqual(nodes["recoveryStatus"]["attrs"].get("role"),"status")
        self.assertEqual(nodes["refreshRecovery"]["tag"],"button")
        self.assertIn("source file remains",html)
        self.assertIn("no mid-video resume",html)
        script=(ROOT / "static/app.js").read_text(encoding="utf-8")
        submit=script.split("async function submitItem(",1)[1].split("async function convertPending(",1)[0]
        self.assertIn('data.append("retain_source",String(frontendContract.retainSourceForSubmission(item)))',submit)


if __name__ == "__main__":
    unittest.main()
