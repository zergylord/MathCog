--[[
    network with a single recurrent hidden layer
    an arbitary number of softmax style action factors
    recurrent connections for the previous sampled actions (not backpropped through)
--]]
require 'nngraph'
require 'nn'
local model = {} 
local num_dim,num_actions
local num_hid = 7
function model.create(n_d,n_act)
    num_dim = n_d
    num_actions = n_act
    local input = nn.Identity()()
    local last_hid = nn.Identity()()
    local pools = {}
    table.insert(pools,input)
    table.insert(pools,last_hid)
    for a=1,(#num_actions)[1] do
        table.insert(pools,nn.Identity()())
    end

    local total_input = nn.JoinTable(2)(pools)
    local hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_actions:sum(),num_hid)(total_input))

    local out_pools = {}
    for a=1,(#num_actions)[1] do
        table.insert(out_pools,nn.LogSoftMax()(nn.Linear(num_hid,num_actions[a])(hid)))
    end
    table.insert(out_pools,hid)

    local network = nn.gModule(pools,out_pools)
    return network
end
local nll_crit = nn.ClassNLLCriterion()
function model.prep_rec_state(mb_size,outputs,actions)
    local act_vec = {}
    for a = 1,(#num_actions)[1] do
        act_vec[a] = torch.zeros(mb_size,num_actions[a])
    end
    if not outputs then --init rec state
        table.insert(act_vec,1,torch.zeros(mb_size,num_hid))
        return act_vec
    else
        for a = 1,(#num_actions)[1] do
            if mb_size > 1 then
                act_vec[a]:scatter(2,actions[a][{{1,mb_size},{}}]:long(),1)
            else
                act_vec[a][1][actions[a][1][1]] = 1
            end
        end
        table.insert(act_vec,1,outputs[3][{{1,mb_size},{}}])
        return act_vec
    end
end

function model.prep_grads(mb_size,outputs,actions,R,prev_grads)
    local grad = {}
    local loss = 0
    for a =1,(#num_actions)[1] do
        loss = loss + nll_crit:forward(outputs[a],actions[a][{{},1}])
        grad[a] = nll_crit:backward(outputs[a],actions[a][{{},1}]):clone()
        grad[a]:cmul(R:repeatTensor(1,grad[a]:size()[2]))
    end
    grad[3] = torch.zeros(mb_size,num_hid)
    if prev_grads then
        grad[3][{{1,prev_grads[2]:size()[1]},{}}] = prev_grads[2]
    end
    return loss,grad
end

function model.sample_actions(outputs)
    actions = {}
    for a = 1,(#num_actions)[1] do
        actions[a] = torch.multinomial(torch.exp(outputs[a]),1)
    end
    return actions
end
return model
