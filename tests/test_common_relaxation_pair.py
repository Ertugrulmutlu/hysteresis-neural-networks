import csv
from pathlib import Path

import torch
import yaml

from src.validate_pair import validate_pair


def make_relaxation_run(path: Path, scenario: str, steps=(0, 2)):
    path.mkdir(); config={"experiment":{"seed":1},"data":{},"model":{},"split":{},
        "train":{"scenario":scenario,"epochs_total":2,"phase_epochs":1,"relaxation_steps":steps[-1],
                 "relaxation_checkpoints":list(steps)},"logging":{"run_name":scenario,"save_dir":"x"}}
    (path/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    state={"x":torch.tensor([1.])}; torch.save(state,path/"weights_epoch_000.pt"); torch.save(state,path/"weights_epoch_001.pt"); torch.save(state,path/"weights_epoch_002.pt")
    phase_data=("A","B") if scenario=="SABC" else ("B","A")
    with (path/"metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["epoch","phase","phase_data","train_loss","test_loss_full","test_acc_full","test_loss_A","test_acc_A","test_loss_B","test_acc_B"]); writer.writeheader()
        for epoch in (1,2): writer.writerow({"epoch":epoch,"phase":epoch,"phase_data":phase_data[epoch-1],"train_loss":1,"test_loss_full":1,"test_acc_full":1,"test_loss_A":1,"test_acc_A":1,"test_loss_B":1,"test_acc_B":1})
    with (path/"relaxation_metrics.csv").open("w",newline="") as handle:
        fields=["scenario","relaxation_step","train_loss_C_recent","test_loss_full","test_acc_full","test_loss_A","test_acc_A","test_loss_B","test_acc_B"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for step in steps: writer.writerow({"scenario":scenario,"relaxation_step":step,"train_loss_C_recent":1,"test_loss_full":1,"test_acc_full":1,"test_loss_A":1,"test_acc_A":1,"test_loss_B":1,"test_acc_B":1})
    for step in steps: torch.save(state,path/f"weights_relax_step_{step:06d}.pt")


def test_common_relaxation_pair_and_two_sabc_rejection(tmp_path):
    make_relaxation_run(tmp_path/"a","SABC"); make_relaxation_run(tmp_path/"b","SBAC")
    assert validate_pair(tmp_path/"a",tmp_path/"b")["valid"]
    config=yaml.safe_load((tmp_path/"b"/"config_resolved.yaml").read_text()); config["train"]["scenario"]="SABC"; (tmp_path/"b"/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    assert not validate_pair(tmp_path/"a",tmp_path/"b")["valid"]


def test_missing_checkpoint_and_mismatched_steps(tmp_path):
    make_relaxation_run(tmp_path/"a","SABC"); make_relaxation_run(tmp_path/"b","SBAC")
    (tmp_path/"b"/"weights_relax_step_000002.pt").unlink()
    assert any("Missing relaxation checkpoint" in error for error in validate_pair(tmp_path/"a",tmp_path/"b")["errors"])
    config=yaml.safe_load((tmp_path/"b"/"config_resolved.yaml").read_text()); config["train"]["relaxation_checkpoints"]=[0,1,2]; (tmp_path/"b"/"config_resolved.yaml").write_text(yaml.safe_dump(config))
    assert any("relaxation-step sequences" in error for error in validate_pair(tmp_path/"a",tmp_path/"b")["errors"])
