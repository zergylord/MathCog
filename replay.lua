require 'nngraph'
require 'optim'
require 'math'
local replay = {}
local replay_table,replay_index,replay_size
function replay.init(r_size)
    replay_size = r_size
    replay_index = 0
    replay_table = {}
end
local worst_val = 1/0
local worst_ind = 1
--adds episode data to table of episodes
--each *_hist var should be a tensor of dim <time,data size>
function replay.add_episode(state_hist,action_hist,prob_hist,reward_hist,target_hist)
    replay_index = (replay_index % replay_size) + 1
    --[[insert the best----
    if #replay_table > replay_index then
        if worst_val > reward_hist:sum() then
            return
        else
            replay_index = worst_ind
        end
    end
    --]]-------------------
    replay_table[replay_index] = {}
    replay_table[replay_index].states = state_hist:clone()
    --actions are factored, and thus in a table
    replay_table[replay_index].actions = {}
    for a = 1,#action_hist do
        replay_table[replay_index].actions[a] = action_hist[a]:clone()
    end
    replay_table[replay_index].probs = prob_hist:clone()
    replay_table[replay_index].targets = {}
    for t = 1,reward_hist:size()[1] do
        if target_hist[t] then
            replay_table[replay_index].targets[t] = {}
            for a = 1,#action_hist do
                replay_table[replay_index].targets[t][a] = target_hist[t][a]:clone()
            end
        end
    end
    --print('hist:',action_hist,prob_hist,reward_hist)
    replay_table[replay_index].rewards = reward_hist:clone()
    replay_table[replay_index].length = state_hist:size()[1]
end

--concatenates selected episodes' data together, s.t. its time indexed 
function replay.get_minibatch(mb_size)
    local episodes = {}
    local indices = torch.zeros(mb_size) 
    local lengths = torch.zeros(mb_size)
    for i=1,mb_size do
        indices[i] = torch.random(#replay_table)
        lengths[i] = -replay_table[indices[i]].length
    end
    --sort recalled memories from longest to shortest
    --thus nth memory at time t is also nth memory at time < t
    local ordering
    _,ordering = lengths:sort()
    indices = indices:index(1,ordering:long())
    for i = 1,mb_size do
        local entry = replay_table[indices[i]]
        for j=1,entry.length do
            if not episodes[j] then
                episodes[j] = {}
                episodes[j].state = entry.states[{{j}}]
                episodes[j].action = {}
                for a=1,#entry.actions do
                    episodes[j].action[a] = entry.actions[a][{{j}}]
                end
                episodes[j].reward = entry.rewards[{{j}}]
                episodes[j].prob = entry.probs[{{j}}]
                episodes[j].taught = torch.zeros(1):byte()
                if entry.targets[j] then
                    episodes[j].taught[1] = 1
                    episodes[j].target = {}
                    for a=1,#entry.actions do
                        episodes[j].target[a] = entry.targets[j][a]
                    end
                end
            else
                episodes[j].state = episodes[j].state:cat(entry.states[{{j}}],1)
                for a=1,#entry.actions do
                    episodes[j].action[a] = episodes[j].action[a]:cat(entry.actions[a][{{j}}],1)
                end
                episodes[j].reward = episodes[j].reward:cat(entry.rewards[{{j}}],1)
                episodes[j].prob = episodes[j].prob:cat(entry.probs[{{j}}],1)
                if entry.targets[j] then
                    episodes[j].taught = episodes[j].taught:cat(torch.ones(1):byte())
                    for a=1,#entry.actions do
                        episodes[j].target[a] = episodes[j].target[a]:cat(entry.targets[j][a],1)
                    end
                else
                    episodes[j].taught = episodes[j].taught:cat(torch.zeros(1):byte())
                end
            end
        end
    end
    return episodes
end
return replay
