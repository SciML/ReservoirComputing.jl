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

rm1 = RandomMapping(6, 10)
rm2 = RandomMapping(6; expansion_size = 10)
rm3 = RandomMapping(; permutations = 6, expansion_size = 10)
@test rm1 == rm2 == rm3
