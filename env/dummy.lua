local env = {}
local state
--return max_steps,num_dim,{num_act1,num_act2...}
function env.get_hyper()
    return 10,1,torch.Tensor{3,5}
end
function env.init()
    state = torch.ones(1,1)
    return state
end
function env.step(actions)
    local term = false
    if torch.random(5) == 1 then
        term = true
    end
    r = torch.rand(1,1)
    return state,r,term
end
return env
