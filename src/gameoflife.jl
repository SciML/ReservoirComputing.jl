abstract type AbstractTwoDimCA end
abstract type AbstractGameOfLife <: AbstractTwoDimCA end

struct GameOfLife{T<:Bool} <: AbstractGameOfLife
    generations::Int
    all_runs::AbstractArray{T}
end


function GameOfLife(initial_state::AbstractArray{T},
        generations::Int) where T<: Bool
    
    height, width = size(initial_state)
    
    all_runs = Array{T}(undef, height, width, generations)
    all_runs[:, :, 1] = initial_state
        
    for g = 2:generations
        for i = 1:height, j = 1:width
            live_neighbours = 0
            
            for ud = (i-1):(i+1), lr = (j-1):(j+1)
                #pbc
                if ud < 1; ud = height-ud; end
                if ud > height; ud = ud-height; end
                if lr < 1; lr = width-lr; end
                if lr > width; lr = lr-width; end
                
                if all_runs[ud, lr, g-1] == 1; live_neighbours+=1; end
            end
            
            all_runs[i, j, g] = apply_standard_rules(all_runs[i, j, g-1], live_neighbours)
        end
    end
    return GameOfLife{T}(generations, all_runs)
end

function apply_standard_rules(state::T, 
        live_neighbours::Int) where T <: Bool
    
    if state == 1 
        if live_neighbours == 3 || live_neighbours == 4
            return 1
        else
            return 0
        end
    elseif state == 0
        if live_neighbours == 3
            return 1
        else
            return 0
        end
    end
end 
