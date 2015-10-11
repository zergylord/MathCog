require 'math'
local env = {}

local image,box_pos,catch_pos
local dims = {40,20}
local box_side = 3
local catch_length = 3 
--return max_steps,num_dim,{num_act1,num_act2...}
function env.get_hyper()
    local max_steps = dims[1]
    local num_dim = dims[1]*dims[2]
    local num_actions = torch.Tensor{3}--left,right,stay
    return max_steps,num_dim,num_actions
end
local function render()
    image = torch.zeros(unpack(dims))
    image[{{box_pos[1],box_pos[1]+box_side-1},
            {box_pos[2],box_pos[2]+box_side-1}}] = 1
    image[{{dims[1]},{catch_pos,catch_pos+catch_length-1}}] = 1
end
function env.init()
    box_pos = {1,torch.random(dims[2]-box_side+1)}
    catch_pos = torch.random(dims[2]-catch_length+1)
    render()
    local state = image:reshape(1,dims[1]*dims[2])
    return state,image
end
function env.force_actions()
    return nil
end
function env.step(actions)
    local term = false
    local act = actions[1][1]
    r = torch.zeros(1,1)
    
    --resolve action
    if act == 1 then--left
       catch_pos = math.max(1,catch_pos-1) 
    elseif act == 2 then--right
       catch_pos = math.min((dims[2]-catch_length+1),catch_pos+1) 
    end--act 3 does nothing
    
    --falling box
    box_pos[1] = box_pos[1] + 1

    --resolve terminal condition
    if (box_pos[1]+box_side-1) == dims[1] then
        term = true
        if catch_pos < (box_pos[2]+box_side-1) and (catch_pos + catch_length-1) > box_pos[2] then
            r[1][1] = 1
        else
            r[1][1] = -1
        end
    end
    render()
    local state = image:reshape(1,dims[1]*dims[2])
    return state,r,term,image
end
return env
