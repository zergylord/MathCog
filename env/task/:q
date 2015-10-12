local task = {}
local add1,add2,max_add
function task.init(max_a,state)
    max_add = max_a
    add1 = torch.random(max_add)
    add2 = torch.random(max_add)
    command = {torch.Tensor{{add1+1}},--1
                torch.Tensor{{max_add*2+1+1+1}},--plus
                torch.Tensor{{add2+1}}}--1
    return state,command
end
function task.step(state,act,say,prev_said)
    local term = false
    local r = torch.Tensor{{0}}
    if say == max_add*2+1+1 and prev_said == (add1+add2) then --done and right
        term = true
        r = torch.Tensor{{1}}
    end
    return r,term
end
return task
