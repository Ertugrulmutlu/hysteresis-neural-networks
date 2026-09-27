import pandas as pd, torch, yaml
from src.validate_pair import validate_pair
def make(path,scenario="SAB",weight=1.,dup=False,lr=.1,wrong_phase=False,epochs=(1,)):
    path.mkdir(); total=max(epochs) if epochs else 1; c={"experiment":{"seed":1},"model":{},"train":{"scenario":scenario,"epochs_total":total,"phase_epochs":1,"lr":lr},"logging":{"run_name":scenario,"save_dir":"x"}}; (path/"config_resolved.yaml").write_text(yaml.safe_dump(c)); torch.save({"x":torch.tensor([weight])},path/"weights_epoch_000.pt"); torch.save({"x":torch.tensor([2.])},path/f"weights_epoch_{total:03d}.pt")
    first="A" if scenario=="SAB" else "B"; second="B" if scenario=="SAB" else "A"; rows=[]
    for epoch in epochs:
        phase=1 if epoch<=1 else 2; phase_data=(first if phase==1 else second); phase_data=("B" if phase_data=="A" else "A") if wrong_phase else phase_data
        rows.append({"epoch":epoch,"phase":phase,"phase_data":phase_data,"train_loss":1,"test_loss_full":1,"test_acc_full":1,"test_loss_A":1,"test_acc_A":1,"test_loss_B":1,"test_acc_B":1})
    pd.DataFrame(rows*(2 if dup else 1)).to_csv(path/"metrics.csv",index=False)
def test_pass_and_init_fail(tmp_path):
    make(tmp_path/"a"); make(tmp_path/"b","SBA"); assert validate_pair(tmp_path/"a",tmp_path/"b")["valid"]; torch.save({"x":torch.tensor([9.])},tmp_path/"b"/"weights_epoch_000.pt"); assert not validate_pair(tmp_path/"a",tmp_path/"b")["valid"]
def test_failures(tmp_path):
    make(tmp_path/"a"); make(tmp_path/"b","SBA",dup=True,lr=.2); (tmp_path/"b"/"weights_epoch_001.pt").unlink(); e=validate_pair(tmp_path/"a",tmp_path/"b")["errors"]; assert any("Disallowed" in x for x in e) and any("Epochs" in x for x in e) and any("final checkpoint" in x for x in e)

def test_scenario_order_and_phase_semantics(tmp_path):
    make(tmp_path/"a","SAB"); make(tmp_path/"b","SAB")
    assert any("run_a=SAB and run_b=SBA" in e for e in validate_pair(tmp_path/"a",tmp_path/"b")["errors"])
    (tmp_path/"b").rename(tmp_path/"old"); make(tmp_path/"b","SBA",wrong_phase=True)
    assert any("Incorrect phase semantics" in e for e in validate_pair(tmp_path/"a",tmp_path/"b")["errors"])

def test_missing_init_row_count_and_non_contiguous_epochs(tmp_path):
    make(tmp_path/"a","SAB",epochs=(1,3)); make(tmp_path/"b","SBA",epochs=(1,3)); (tmp_path/"b"/"weights_epoch_000.pt").unlink()
    errors=validate_pair(tmp_path/"a",tmp_path/"b")["errors"]
    assert any("Missing initialization" in e for e in errors)
    assert any("row count" in e for e in errors)
    assert any("exactly 1..epochs_total" in e for e in errors)
