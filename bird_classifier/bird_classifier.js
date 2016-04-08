// BIRD CLASSIFIER
// Defines and runs a pre-trained deep convolutional network to perform bird classification.
// Called from bird_classification.html


// set constants

var IMAGE_DIMENSION = 227;
var IMAGE_CHANNELS = 3;



// define the network architecture (AlexNet)

layer_defs = [];

layer_defs.push({type:'input', out_sx:227, out_sy:227, out_depth:3});
layer_defs.push({type:'conv', sx:11, filters:96, stride:4, pad:0, activation:'relu'});
layer_defs.push({type:'lrn', k:0, n:5, alpha:0.0001, beta:0.75});
layer_defs.push({type:'pool', sx:3, stride:2});
layer_defs.push({type:'conv', sx:5, filters:256, stride:1, pad:2, activation:'relu'});
layer_defs.push({type:'lrn', k:0, n:5, alpha:0.0001, beta:0.75});
layer_defs.push({type:'pool', sx:3, stride:2});
layer_defs.push({type:'conv', sx:3, filters:384, stride:1, pad:1, activation:'relu'});
layer_defs.push({type:'conv', sx:3, filters:384, stride:1, pad:1, activation:'relu'});
layer_defs.push({type:'conv', sx:3, filters:256, stride:1, pad:1, activation:'relu'});
layer_defs.push({type:'pool', sx:3, stride:2});
layer_defs.push({type:'fc', num_neurons:4096, activation: 'relu'});
layer_defs.push({type:'dropout', drop_prob:0.5});
layer_defs.push({type:'fc', num_neurons:4096, activation: 'relu'});
layer_defs.push({type:'dropout', drop_prob:0.5});
layer_defs.push({type:'softmax', num_classes:555});

net = new convnetjs.Net();
net.makeLayers(layer_defs);



// load the network weights




var x = convnetjs.img_to_vol(document.getElementById('input_image'))
var output_probabilities_vol = net.forward(x)








