require 'nngraph'
require 'optim'
require 'math'
replay = require 'replay'

replay.init(100)
for i=1,150 do
    length = torch.random(10)
    state_hist = torch.rand(length,10)
    action_hist = torch.rand(length,3)
    reward_hist = torch.rand(length,1)
    replay.add_episode(state_hist,action_hist,reward_hist)
end
a =replay.get_minibatch(32)
print(a)
