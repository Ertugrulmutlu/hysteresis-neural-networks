import csv, json, pytest, torch, yaml
from src.tracker import Tracker
def cfg(): return {"experiment":{"seed":1,"device":"cpu"},"train":{"scenario":"SAB"}}
def test_tracker(tmp_path):
    run=tmp_path/"run"; t=Tracker(run,cfg()); assert yaml.safe_load((run/"config_resolved.yaml").read_text())==cfg()
    t.log_metrics({"epoch":1,"loss":2}); t.log_metrics({"epoch":2,"loss":1}); assert len((run/"metrics.csv").read_text().splitlines())==3
    assert t.save_weights(torch.nn.Linear(2,1),0).name=="weights_epoch_000.pt" and json.loads((run/"run_metadata.json").read_text())["seed"]==1
    with pytest.raises(FileExistsError): Tracker(run,cfg())
    Tracker(run,cfg(),overwrite=True); assert not (run/"weights_epoch_000.pt").exists()

def test_relaxation_checkpoint_and_metrics(tmp_path):
    tracker=Tracker(tmp_path/"relax",cfg()); model=torch.nn.Linear(2,1)
    assert tracker.save_relaxation_weights(model,100).name=="weights_relax_step_000100.pt"
    for step in (0,100): tracker.log_relaxation_metrics({"scenario":"SABC","relaxation_step":step})
    rows=list(csv.DictReader((tmp_path/"relax"/"relaxation_metrics.csv").open()))
    assert [int(row["relaxation_step"]) for row in rows]==[0,100]
