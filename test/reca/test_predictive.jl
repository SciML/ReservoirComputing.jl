using ReservoirComputing, CellularAutomata

const input = ones(2, 10)
const output = zeros(2, 10)
const g = 6
const rule = 90

reca = RECA(input, DCA(rule); 
    generations = g,
    input_encoding = RandomMapping(6, 10))

output_layer = train(reca, output, StandardRidge(0.001))
prediction = reca(Predictive(input), output_layer)
final_pred = convert(AbstractArray{Int}, prediction .> 0.5)
@test final_pred == output