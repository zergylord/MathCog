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
model = require 'model.teach_rec_softmax_RE'
local profile = false
if profile then
    ProFi = require 'ProFi'
    ProFi:start()
end
--env = require 'env.bandit'
env = require 'env.token'
--env = require 'env.catch'
local max_steps,num_dim,num_actions = env.get_hyper()
network = model.create(num_dim,num_actions)
w,dw = network:getParameters()
network:zeroGradParameters()
local net_clones = util.clone_many_times(network,max_steps)

local timer = torch.Timer()
local mb_size = 32
local replay_size = 100000
local burn_in = 500
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
        local cur_size = data[t].state:size()[1]
        if t>1 and model.prep_td then
            model.prep_td(data[t-1].reward[{{1,cur_size}}],outputs)
        end
        if t+1 <= last_step then
            rec_state = model.prep_rec_state(data[t+1].state:size()[1],outputs,data[t].action)
        end
        state_hist[t] = state
        output_hist[t] = outputs
    end
    --backward pass through episode
    loss = model.backward(net_clones,mb_size,last_step,state_hist,output_hist,data)
    --clip gradients
    --dw:clamp(-1,1)
    return loss,dw
end
local optim_state = {learningRate = 1e-4}
local max_iter = 1e6
local net_loss = 0
local r = 0
local cum = 0
for iter = 1,max_iter do
    local state_hist = torch.zeros(max_steps,num_dim)
    local action_hist = {}
    local target_hist = {}
    for a =1,(#num_actions)[1] do
        action_hist[a] = torch.zeros(max_steps,1)
    end
    local prob_hist = torch.zeros(max_steps,(#num_actions)[1])
    local reward_hist = torch.zeros(max_steps,1)
    local t = 0
    local term = false
    local state = env.init()
    local rec_state = model.prep_rec_state(1)

    local  print_q = false
    --[[
    if torch.rand(1)[1] < .01 then
        print_q = true
        print('start:')
    end
    --]]
    local print_actions = false
    if torch.rand(1)[1] < .002 then
        print_actions = true
        print('start:')
    end
    while t<max_steps and not term do
        t = t + 1
        local total_state = {state}
        for e = 1,#rec_state do
            table.insert(total_state,rec_state[e])
        end
        local outputs = network:forward(total_state)
        if print_q then
            print(outputs[4][1][1])
        end
        local actions,probs = model.sample_actions(outputs)
        prob_hist[t] = probs:clone()
        if env.force_actions then
            actions = env.force_actions() or actions
        end
        if print_actions then
            print(actions[2][1][1],probs[2])
        end
        rec_state = model.prep_rec_state(1,outputs,actions)

        state_hist[t] = state
        --[[
        local teach_step = false
        if torch.rand(1)[1] < .1 then
            teach_step = true
        end
        if teach_step then
            target_hist[t] = {}
        end
        -]]
        for a=1,(#num_actions)[1] do
            action_hist[a][t] = actions[a] 
            --[[
            if teach_step then
                target_hist[t][a] = actions[a] 
            end
            -]]
        end
        state,r,term,target_hist[t] = env.step(actions)
        reward_hist[t] = r
        cum = cum + r[1][1]
    end
    for a =1,(#num_actions)[1] do
        action_hist[a] = action_hist[a][{{1,t},{}}]
    end
    --local target_hist = action_hist
    replay.add_episode(state_hist[{{1,t},{}}],
                        action_hist,prob_hist[{{1,t},{}}],
                        reward_hist[{{1,t},{}}],
                        target_hist)
    if iter >= burn_in then
        _,cur_loss = optim.rmsprop(feval,w,optim_state)
        net_loss = net_loss + cur_loss[1]
        if iter % 1000 == 0 then
            print(iter,cum,net_loss,dw:norm(),w:norm(),timer:time().real)
            net_loss = 0
            cum = 0
            timer:reset()
            collectgarbage()
        end
    end
end
if profile then
    ProFi:stop()
    ProFi:writeReport('train_report.txt')
end
