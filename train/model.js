const tf = require('@tensorflow/tfjs');

const kernelSize = [3, 3];
const poolSize= [2, 2];
const firstFilters = 32;

const model = tf.sequential();

model.add(tf.layers.conv2d({
  inputShape: [96, 96, 3],
  filters: firstFilters,
  kernelSize: kernelSize,
  activation: 'relu',
}));

model.add(tf.layers.maxPooling2d({
  poolSize: poolSize
}));

model.add(tf.layers.conv2d({
  filters: firstFilters,
  kernelSize: kernelSize,
  activation: 'relu',
}));

model.add(tf.layers.maxPooling2d({
  poolSize: poolSize
}));


model.add(tf.layers.flatten());

model.add(tf.layers.dense({
  units: 128,
  activation: 'relu',
  useBias: true
}));

model.add(tf.layers.dense({
  units: 101,
  activation: 'softmax'
}));

const optimizer = tf.train.adam(0.0001);
model.compile({
  optimizer,
  loss: 'binaryCrossentropy',
  metrics: ['accuracy'],
});

module.exports = model;
