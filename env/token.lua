local env = {}
--local task = require 'env.task.add'
local task = require 'env.task.count_list'

local max_add = 10
--add token, subtract token, do nothing
local num_act = 3
--count range + done + command words ('plus','how many?','give','tokens')
local num_say = max_add*2+1+1+4 

local state,command,command_ind,prev_said,token_index
--return max_steps,num_dim,{num_act1,num_act2...}
function env.get_hyper()
    local max_steps = max_add*2*3
    local num_dim = max_add*2 --workspace
    local num_actions = torch.Tensor{num_act,num_say} --act,say
    return max_steps,num_dim,num_actions
end
function env.init()
    state = torch.zeros(1,max_add*2)
    state,command = task.init(max_add,state)
    command_ind = 1
    token_ind = 1--where next token goes
    prev_said = -1
    return state
end
function env.force_actions()
    if command_ind <= #command then
        local actions = {torch.Tensor{{torch.random(num_act)}},command[command_ind]}
        command_ind = command_ind + 1
        return actions
    else
        return nil
    end
end
function env.step(actions)
    local term = false
    local act = actions[1][1][1]
    local say = actions[2][1][1]
    r,term = task.step(state[1],act,say,prev_said)
    if not term then
        prev_said = say -1
        if act == 1 then --add token
            state[1][token_ind] = 1
            token_ind = math.min(max_add*2,token_ind + 1)
        elseif act == 2 then --remove token
            token_ind = math.max(1,token_ind - 1)
            state[1][token_ind] = 0
        end
    end
    return state,r,term
end
return env
