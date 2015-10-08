--[[

    supports somewhat arbitary network architecture
    -assumes 1 state input, and an arbitary number of additional inputs
    corresponding to recurrent state information, i.e. a function of the
    outputs and actions prep_rec_state(outputs,actions)
    -assume an arbitary number of outputs, but forces 
--]]
util = require 'util.model_utils'
require 'nngraph'
require 'optim'
require 'math'
replay = require 'replay'
model = require 'model.basic_rec_act'
env = require 'env.dummy'
local max_steps,num_dim,num_actions = env.get_hyper()
network = model.create(num_dim,num_actions)
w,dw = network:getParameters()
network:zeroGradParameters()
local net_clones = util.clone_many_times(network,max_steps)


local mb_size = 32
local replay_size = 100
replay.init(replay_size)
--[[
for i=1,150 do
    length = torch.random(10)
    state_hist = torch.rand(length,10)
    action_hist = {torch.ones(length,1),torch.ones(length,1)}
    reward_hist = torch.rand(length,1)
    replay.add_episode(state_hist,action_hist,reward_hist)
end
a =replay.get_minibatch(32)
--]]
function feval(x)
    if x ~= w then
        w:copy(x)
    end
    dw:zero()
    local loss = 0
    local data = replay.get_minibatch(mb_size)
    local rec_state = model.prep_rec_state(mb_size)
    local last_step = #data
    local state_hist = {}
    local output_hist = {}
    local reward_hist = {}
    --forward pass through episode
    for t = 1,last_step do
        local state = {data[t].state}
        for e = 1,#rec_state do
            table.insert(state,rec_state[e])
        end
        local outputs = net_clones[t]:forward(state)
        if t+1 <= last_step then
            rec_state = model.prep_rec_state(data[t+1].state:size()[1],outputs,data[t].action)
        end
        state_hist[t] = state
        output_hist[t] = outputs
        reward_hist[t] = data[t].reward
    end
    --backward pass through episode
    local R = torch.zeros(mb_size,1) 
    local prev_grad
    for t = last_step,1,-1 do
        local cur_size = reward_hist[t]:size()[1]
        R[{{1,cur_size}}] = R[{{1,cur_size}}] + reward_hist[t]
        local grad
        loss,grad = model.prep_grads(cur_size,output_hist[t],data[t].action,R[{{1,cur_size}}],prev_grad)
        prev_grad = net_clones[t]:backward(state_hist[t],grad)
    end  
    return loss,dw
end
local optim_state = {learningRate = 1e-4}
local max_iter = 1e4
local net_loss = 0
local r = 0
local state = env.init()
local rec_state = model.prep_rec_state(1)
for iter = 1,max_iter do
    local state_hist = torch.zeros(max_steps,num_dim)
    local action_hist = {}
    for a =1,(#num_actions)[1] do
        action_hist[a] = torch.zeros(max_steps,1)
    end
    local reward_hist = torch.zeros(max_steps,1)
    local t = 0
    local term = false
    while t<max_steps and not term do
        t = t + 1
        local total_state = {state}
        for e = 1,#rec_state do
            table.insert(total_state,rec_state[e])
        end
        local outputs = network:forward(total_state)
        actions = model.sample_actions(outputs)
        rec_state = model.prep_rec_state(1,outputs,actions)

        state_hist[t] = state
        for a=1,(#num_actions)[1] do
            action_hist[a][t] = actions[a] 
        end
        state,r,term = env.step(actions)
        reward_hist[t] = r
    end
    for a =1,(#num_actions)[1] do
        action_hist[a] = action_hist[a][{{1,t},{}}]
    end
    replay.add_episode(state_hist[{{1,t},{}}],action_hist,reward_hist[{{1,t},{}}])
    if iter >= mb_size then
        _,net_loss = optim.rmsprop(feval,w,optim_state)
        print(net_loss[1])
    end
end
