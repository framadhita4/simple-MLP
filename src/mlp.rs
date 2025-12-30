use ndarray::Array2;
use rand::Rng;

use crate::autograd::Autograd;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
    None,
}

pub struct Neuron {
    weights: Vec<Autograd>,
    bias: Autograd,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();

        let scale = (2.0 / nin as f64).sqrt();
        let weights = (0..nin)
            .map(|_| Autograd::new(Array2::from_elem((1, 1), rng.gen_range(-scale..scale))))
            .collect();
        let bias = Autograd::new(Array2::from_elem((1, 1), 0.0));

        Self { weights, bias }
    }

    pub fn call(&self, x: &[Autograd], activation: Activation) -> Autograd {
        let mut sum = self.bias.clone();
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            // sum = sum + w * xi
            sum = sum.add(&w.mul(xi));
        }

        match activation {
            Activation::ReLU => sum.relu(),
            Activation::Tanh => sum.tanh(),
            Activation::None => sum,
        }
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
    activation: Activation,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, activation: Activation) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin)).collect();
        Self {
            neurons,
            activation,
        }
    }

    pub fn call(&self, x: &[Autograd]) -> Vec<Autograd> {
        self.neurons
            .iter()
            .map(|n| n.call(x, self.activation))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);

        let layers = (0..nouts.len())
            .map(|i| {
                let activation = if i < nouts.len() - 1 {
                    Activation::ReLU
                } else {
                    Activation::None
                };
                Layer::new(sizes[i], sizes[i + 1], activation)
            })
            .collect();

        Self { layers }
    }

    pub fn call(&self, x: &[Autograd]) -> Vec<Autograd> {
        let mut current = x.to_vec();
        for layer in &self.layers {
            current = layer.call(&current);
        }
        current
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
