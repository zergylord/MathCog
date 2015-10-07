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
--each *_hist var should be a tensor of dim <time,data size>
function replay.add_episode(state_hist,action_hist,reward_hist)
    replay_index = (replay_index % replay_size) + 1
    replay_table[replay_index] = {}
    replay_table[replay_index].states = state_hist:clone()
    replay_table[replay_index].actions = action_hist:clone()
    replay_table[replay_index].rewards = reward_hist:clone()
    replay_table[replay_index].length = state_hist:size()[1]
end

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
        print(entry.rewards)
        for j=1,entry.length do
            if not episodes[j] then
                episodes[j] = {}
                episodes[j].state = entry.states[{{j}}]
                episodes[j].action = entry.actions[{{j}}]
                episodes[j].reward = entry.rewards[{{j}}]
            else
                episodes[j].state = episodes[j].state:cat(entry.states[{{j}}],1)
                episodes[j].action = episodes[j].action:cat(entry.actions[{{j}}],1)
                episodes[j].reward = episodes[j].reward:cat(entry.rewards[{{j}}],1)
            end
        end
    end
    return episodes
end
return replay
