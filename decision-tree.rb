require "rubygems"
require "bundler/setup"
require 'csv'

Bundler.require

training = Ai4r::Data::DataSet.new.load_csv_with_labels('data/training_data_with_labels.csv')
id3 = Ai4r::Classifiers::ID3.new.build(training)

p id3.get_rules

tests = 
 [[0,   0,    0,  0,    0,    1,    1,  0],
  [1,   0,    1,  0,    1,    0,    3,  0],
  [0,   1,    1,  3,    0,    1,    4,  1],
  [0,   0,    0,  0.5,  1,    0,    2,  0],
  [0,   0,    0,  0.4,  0.5,  0,    2,  0],
  [0.1, 0,    0,  0,    0.5,  0,    1,  0],
  [0,   0.5,  0,  0,    0.0,  0,    1,  0],
  [0,   0,    50, 0,    0.0,  0,    1,  0],
  [0,   0,    0,  15,   0.0,  0,    1,  0],
  [0,   0,    0,  0,    100,  0,    1,  0],
  [0,   1,    1,  1,    1,    0,    4,  1],
  [1,   1,    1,  0,    2,    0,    4,  1],
  [0,   1,    1,  1,    2,    1,    5,  1]]

tests.each do |test|
 decision = id3.eval(test[0..-2])
 puts "Predicted: #{decision} ... True decision: #{test.last}"
end

# Error: undefined local variable or method `rule_not_found'