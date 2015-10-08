require 'nngraph'
require 'nn'
 --TODO:set by task----------
local num_dim = 10
local num_act = 3 
local num_say = 5
-----------------------------
local num_hid = 7
local input = nn.Identity()()
local last_hid = nn.Identity()()
local last_act = nn.Identity()()
local last_say = nn.Identity()()
local total_input = nn.JoinTable(2)({input,last_hid,last_act,last_say})
local hid = nn.ReLU()(nn.Linear(num_dim+num_hid+num_act+num_say,num_hid)(total_input))
local action = nn.LogSoftMax()(nn.Linear(num_hid,num_act)(hid))
local speak = nn.LogSoftMax()(nn.Linear(num_hid,num_say)(hid))
network = nn.gModule({input,last_hid,last_act,last_say},{action,speak,hid})
local model = {} 
local nll_crit = nn.ClassNLLCriterion()
function model.prep_rec_state(mb_size,outputs,actions)
    local act_vec = torch.zeros(mb_size,num_act)
    local say_vec = torch.zeros(mb_size,num_say)
    if not outputs then --init rec state
        return {torch.zeros(mb_size,num_hid),
                act_vec,
                say_vec}
    else
        act_vec:scatter(2,actions[1][{{1,mb_size},{}}]:long(),1)
        say_vec:scatter(2,actions[2][{{1,mb_size},{}}]:long(),1)
        return {outputs[3][{{1,mb_size},{}}],
                act_vec,
                say_vec}
    end
end

function model.prep_grads(mb_size,outputs,actions,R,prev_grads)
    local grad = {}
    local loss = 0
    for a =1,2 do
        loss = loss + nll_crit:forward(outputs[a],actions[a][{{},1}])
        grad[a] = nll_crit:backward(outputs[a],actions[a][{{},1}]):clone()
        grad[a]:cmul(R:repeatTensor(1,grad[a]:size()[2]))
    end
    grad[3] = torch.zeros(mb_size,num_hid)
    if prev_grads then
        grad[3][{{1,prev_grads[2]:size()[1]},{}}] = prev_grads[2]
    end
    return grad
end

return model
