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
local base_lr = .5 --percent of learning rate for baseline
function model.create(n_d,n_act)
    num_dim = n_d
    num_actions = n_act
    act_factors = (#num_actions)[1] 
    local input = nn.Identity()()
    local last_hid = nn.Identity()()
    local in_pools = {}
    local to_join = {}
    table.insert(in_pools,input)
    table.insert(to_join,input)
    table.insert(in_pools,last_hid)
    table.insert(to_join,last_hid)
    local act_input = {}
    for a=1,act_factors do
        act_input[a] = nn.Identity()()
        table.insert(in_pools,act_input[a])
        table.insert(to_join,act_input[a])
        --table.insert(to_join,nn.ReLU()(nn.Linear(num_actions[a],num_hid)(act_input[a])))
    end

    local total_input = nn.JoinTable(2)(to_join)
    local hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_actions:sum(),num_hid)(total_input))
    --local hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_hid*act_factors,num_hid)(total_input))

    local out_pools = {}
    for a=1,act_factors do
        table.insert(out_pools,nn.LogSoftMax()(nn.Linear(num_hid,num_actions[a])(hid)))
    end
    table.insert(out_pools,hid)
    local baseline = nn.Linear(num_hid,1)(hid)
    table.insert(out_pools,baseline)

    local network = nn.gModule(in_pools,out_pools)
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
        local one_hot_val = 1
        for a = 1,act_factors do
            if mb_size > 1 then
                act_vec[a]:scatter(2,actions[a][{{1,mb_size},{}}]:long(),one_hot_val)
            else
                act_vec[a][1][actions[a][1][1]] = one_hot_val
            end
        end
        --IVE CUT  THE RECURRENT CONNECtION
        table.insert(act_vec,1,torch.zeros(mb_size,num_hid))
        --table.insert(act_vec,1,outputs[act_factors+1][{{1,mb_size},{}}])
        return act_vec
    end
end

local nll_crit = nn.ClassNLLCriterion()
local mse_crit = nn.MSECriterion()
--TODO:implement these
local td_base = true
local gamma = .7
function model.backward(net_clones,mb_size,last_step,states,outputs,data)
    local R = torch.zeros(mb_size,1) 
    local prev_grad
    local loss = 0
    for t = last_step,1,-1 do
        local cur_size = data[t].reward:size()[1]
        --TODO:replace with actual taught field
        data[t].taught = torch.rand(cur_size,1):gt(.05)
        R[{{1,cur_size}}] = R[{{1,cur_size}}] + data[t].reward
        local grad = {}
        local b = outputs[t][act_factors+2]
        --action grad-------
        for a =1,act_factors do
            loss = loss + nll_crit:forward(outputs[t][a],data[t].action[a][{{},1}])
            grad[a] = nll_crit:backward(outputs[t][a],data[t].action[a][{{},1}]):clone()
            grad[a]:cmul((R[{{1,cur_size}}]-b):repeatTensor(1,grad[a]:size()[2]))
            --importance sampling
            local cur_prob = outputs[t][a]:gather(2,data[t].action[a]:long()):exp()
            cur_prob:cdiv(data[t].prob[{{},a}]):cmin(10) --truncated IS
            grad[a]:cmul(cur_prob:repeatTensor(1,grad[a]:size()[2]))
            --]]
            --supervised part of grad
            local ele_mask = data[t].taught
            local num_masked = ele_mask:sum()
            if num_masked ~= 0 then
                local act_size = outputs[t][a]:size()[2]
                local act_mask = data[t].taught:repeatTensor(1,act_size)
                local masked_out = outputs[t][a][act_mask]:reshape(num_masked,act_size)
                local act_vec = torch.zeros(num_masked,act_size)
                local masked_target = act_vec:scatter(2,
                    data[t].action[a][ele_mask]:reshape(num_masked,1):long(),1)
                loss = loss + mse_crit:forward(masked_out,masked_target)
                grad[a][act_mask] = grad[a][act_mask] + mse_crit:backward(masked_out,masked_target) 
            end
        end
        --recurrent
        grad[act_factors+1] = torch.zeros(cur_size,num_hid)
        if prev_grads then
            grad[act_factors+1][{{1,prev_grads[2]:size()[1]},{}}] = prev_grads[2]
        end
        --baseline
        loss = loss + mse_crit:forward(b,R[{{1,cur_size}}])
        grad[act_factors+2] = mse_crit:backward(b,R[{{1,cur_size}}]):mul(base_lr)
        prev_grad = net_clones[t]:backward(states[t],grad)
    end
    return loss
end

function model.sample_actions(outputs)
    actions = {}
    probs = torch.zeros(act_factors)
    for a = 1,act_factors do
        actions[a] = torch.multinomial(torch.exp(outputs[a]),1)
        probs[a] = torch.exp(outputs[a][{{},actions[a][1][1]}])
    end
    return actions,probs
end
return model
