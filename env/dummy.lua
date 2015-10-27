local env = {}
local state
local count = 0
--return max_steps,num_dim,{num_act1,num_act2...}
function env.get_hyper()
    return 10,1,torch.Tensor{3,5}
end
function env.init()
    state = torch.ones(1,1)
    return state
end
function env.step(actions)
    count = count + 1
    local term = false
    r = torch.zeros(1,1)
    if count == 5 then
        count = 0
        term = true
        r = torch.ones(1,1)
    end
    return state,r,term
end
return env
