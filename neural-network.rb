require "rubygems"
require "bundler/setup"
require 'csv'

Bundler.require

training = []
results = []

raw_training = File.open('data/training_data.csv').read
raw_training.gsub!(/\r\n?/, "\n")
10.times do
  raw_training.each_line{|l| line = l.split(','); training << line[0..-2].map{|f| f.to_i}; results << [line.last.to_i]}
end

trained = RubyFann::TrainData.new(:inputs=>training, :desired_outputs=>results)
fann = RubyFann::Standard.new(:num_inputs=>7, :hidden_neurons=>[7, 2, 9, 2, 4], :num_outputs=>1)

fann.train_on_data(trained, 10000, 10, 0.1) # 50000 max_epochs, 10 errors between reports and 0.1 desired MSE (mean-squared-error)

#p id3.get_rules

tests = 
 [[0,   0,    0,  0,    0,    1,    1,  0],
  [1,   0,    1,  0,    1,    0,    3,  0],
  [0,   1,    1,  3,    0,    1,    4,  1],
  [0,   0,    0,  0.5,  1,    0,    2,  0],
  [0,   0,    0,  0.4,  0.5,  0,    2,  0],
  [0.1, 0,    0,  0,    0.5,  0,    1,  0],
  [0,   0.5,  0,  0,    0,    0,    1,  0],
  [0,   0,    50, 0,    0,    0,    1,  0],
  [0,   0,    0,  15,   0,    0,    1,  0],
  [0,   0,    0,  0,    100,  0,    1,  0],
  [0,   1,    1,  1,    1,    0,    4,  1],
  [1,   1,    1,  0,    2,    0,    4,  1],
  [0,   1,    1,  1,    2,    1,    5,  1],
  [1,   0,    1,  0,    2,    1,    4,  1]]

tests.each do |test|
  decision = fann.run(test[0..-2])
  puts "Predicted: #{decision} ... True decision: #{test.last}"
end