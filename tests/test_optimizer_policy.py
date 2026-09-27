import pytest
import torch
from src.optimizer import apply_relaxation_optimizer_policy, build_optimizer
from src.train import initialize_relaxation

def config(policy): return {"optimizer":"sgd","lr":.1,"momentum":.9,"weight_decay":0.,"relaxation_optimizer_policy":policy}
def test_preserve_and_reset_optimizer_state_without_weight_change():
    model=torch.nn.Linear(2,1); optimizer=build_optimizer(model.parameters(),config("preserve")); model(torch.ones(1,2)).sum().backward();optimizer.step()
    assert apply_relaxation_optimizer_policy(optimizer,model,config("preserve")) is optimizer
    weights={k:v.clone() for k,v in model.state_dict().items()};reset=apply_relaxation_optimizer_policy(optimizer,model,config("reset"))
    assert reset is not optimizer and not reset.state and all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError): apply_relaxation_optimizer_policy(optimizer,model,config("invalid"))

def test_step_zero_is_saved_before_reset():
    events=[]
    class Tracker:
        def save_relaxation_weights(self,model,step): events.append(("save",step,{k:v.clone() for k,v in model.state_dict().items()}))
    model=torch.nn.Linear(2,1);optimizer=build_optimizer(model.parameters(),config("reset"));replacement=initialize_relaxation(Tracker(),model,optimizer,config("reset"))
    assert events[0][0:2]==("save",0) and replacement is not optimizer
