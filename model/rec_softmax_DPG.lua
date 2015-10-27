--[[
    network with a single recurrent hidden layer
    an arbitary number of softmax style action factors
    recurrent connections for the previous sampled actions (not backpropped through)
--]]
require 'nngraph'
require 'nn'
local model = {} 
local num_dim,num_actions,act_factors
local num_hid = 7
local gamma = 1
local use_td = true
function model.create(n_d,n_act)
    num_dim = n_d
    num_actions = n_act
    act_factors = (#num_actions)[1] 
    local input = nn.Identity()()
    local last_hid = nn.Identity()()
    local pools = {}
    table.insert(pools,input)
    table.insert(pools,last_hid)
    --actor---------------------
    for a=1,act_factors do
        table.insert(pools,nn.Identity()())
    end

    local total_input = nn.JoinTable(2)(pools)
    local hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_actions:sum(),num_hid)(total_input))

    local out_pools = {}
    local act_pools = {}
    for a=1,act_factors do
        local action = nn.SoftMax()(nn.Linear(num_hid,num_actions[a])(hid))
        --local action = nn.Linear(num_hid,num_actions[a])(hid)
        table.insert(out_pools,action)
        table.insert(act_pools,action)
    end
    table.insert(out_pools,hid)
    --critic-----------------------
    local all_actions
    if act_factors > 1 then
        all_actions = nn.JoinTable(2)(act_pools)
    else
        all_actions = act_pools[1]
    end
    local state_and_action = nn.JoinTable(2)({total_input,all_actions})
    local critic_hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_actions:sum()*2,num_hid)(state_and_action))
    local q = nn.Linear(num_hid,1)(critic_hid)
    table.insert(out_pools,q)

    local network = nn.gModule(pools,out_pools)
    return network
end
function model.prep_rec_state(mb_size,outputs,actions)
    local act_vec = {}
    for a = 1,act_factors do
        act_vec[a] = torch.zeros(mb_size,num_actions[a])
    end
    if not outputs then --init rec state
        table.insert(act_vec,1,torch.zeros(mb_size,num_hid))
        return act_vec
    else
        for a = 1,act_factors do
            if mb_size > 1 then
                act_vec[a]:scatter(2,actions[a][{{1,mb_size},{}}]:long(),1)
            else
                act_vec[a][1][actions[a][1][1]] = 1
            end
        end
        table.insert(act_vec,1,outputs[act_factors+1][{{1,mb_size},{}}])
        return act_vec
    end
end
local mse_crit = nn.MSECriterion()
function model.prep_grads(net_clones,mb_size,last_step,states,outputs,data)
    local R = torch.zeros(mb_size,1) 
    local prev_grad
    local loss = 0
    local cur_size = 0
    local print_q_target = false
    --[[
    if torch.rand(1)[1] < .001 then
        print_q_target = true
        print('targets:')
    end
    --]]
    for t = last_step,1,-1 do
        local new_term = data[t].reward:size()[1] - cur_size
        cur_size = data[t].reward:size()[1]
        R[{{1,cur_size}}] = R[{{1,cur_size}}] + data[t].reward
        local grad = {}
        --TODO: cache this; act outpools recieve no external gradients
        for a =1,act_factors do
            grad[a] = torch.zeros(outputs[t][a]:size())
        end
        local q = outputs[t][act_factors+2]
        local q_target = data[t].reward:clone()
        if new_term ~= cur_size then --some non-terminal
            q_target[{{1,cur_size-new_term},{}}]:
                add(outputs[t+1][act_factors+2][{{1,cur_size-new_term},{}}]:clone():mul(gamma))
        end
        if print_q_target then
            print(q_target)
        end

        --recurrent
        grad[act_factors+1] = torch.zeros(cur_size,num_hid)
        if prev_grads then
            grad[act_factors+1][{{1,cur_size},{}}] = prev_grads[2][{{1,cur_size},{}}]
        end
        if use_td then
            loss = loss + mse_crit:forward(q,q_target)
            grad[act_factors+2] = mse_crit:backward(q,q_target)
        else
            loss = loss + mse_crit:forward(q,R[{{1,cur_size}}])
            grad[act_factors+2] = mse_crit:backward(q,R[{{1,cur_size}}])
        end
        prev_grad = net_clones[t]:backward(states[t],grad)
    end
    return loss
end
local softmax = nn.SoftMax()
function model.sample_actions(outputs)
    actions = {}
    probs = torch.zeros(act_factors)
    for a = 1,act_factors do
        local transformed = outputs[a]
        --local transformed = softmax:forward(outputs[a])
        --local transformed = softmax:forward(outputs[a] + torch.randn(outputs[a]:size()):mul(1) )
        actions[a] = torch.multinomial(transformed,1)
        --_,actions[a] = (transformed + torch.randn(transformed:size()):mul(2) ):max(2)
        probs[a] = 1
    end
    return actions,probs
end
return model
