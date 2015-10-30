local env = {}
local state
local count
--return max_steps,num_dim,{num_act1,num_act2...}
function env.get_hyper()
    return 1,1,torch.Tensor{2}
end
function env.init()
    state = torch.ones(1,1)
    return state
end
function env.step(actions)
    local term = true
    local r = torch.zeros(1,1)
    r[1][1] = actions[1][1][1][1] - actions[1][1][1][2]
    return state,r,term
end
return env
