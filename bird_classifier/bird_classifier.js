// BIRD CLASSIFIER
// Defines and runs a pre-trained deep convolutional network to perform bird classification.
// Called from bird_classification.html


// set constants

var image_dimension = 227;
var image_channels = 3;



// define the network architecture (AlexNet)

layer_defs = [];

layer_defs.push({type:'input', out_sx:image_dimension, out_sy:image_dimension, out_depth:image_channels});
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
layer_defs.push({type:'softmax', num_classes:1000});


net = new convnetjs.Net();
net.makeLayers(layer_defs);



// load the network weights






// test the image

var testImage = function(img){
    
    // load the image and pass it through the network
    var x = convnetjs.img_to_vol(img)
    var output_prob = net.forward(x)
    
    // get predictions
    var preds =[]
    for(var k=0;k<output_prob.w.length;k++) {
        preds.push({k:k,p:output_prob.w[k]});
    }
    
    preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
    
    // add predictions
    var div = document.document.getElementById('predictions_plot');
    //div.className = 'testdiv';
    
    var probsdiv = document.createElement('div');
    
    var bars = '';
    for(var k=0;k<10;k++) {
        var bar_color = 'rgb(187,85,85)';
        bars += '<div class=\"pp\" style=\"width:' + Math.floor(preds[k].p/1*100) + 'px; background-color:' + bar_color + ';\"></div>'
    }
    
    probsdiv.innerHTML = bars;
    probsdiv.className = 'probsdiv';
    div.appendChild(probsdiv);
    
    //$(div).prependTo($("#predictions_plot")).hide().fadeIn('slow').slideDown('slow');
    $("#top_predictions").text('Top Predictions: ' + preds[0].p);
    
}








