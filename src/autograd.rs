use ndarray::Array2;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
enum Op {
    Add,
    Mul,
    Tanh,
    ReLU,
    None,
}

// Inner data structure
struct AutogradData {
    value: Array2<f64>,
    grad: Array2<f64>,
    children: Vec<Autograd>,
    op: Op,
    backward: Option<fn(&AutogradData)>,
}

// Wrapper with Rc for shared ownership
pub struct Autograd {
    data: Rc<RefCell<AutogradData>>,
}
impl Autograd {
    pub fn new(value: Array2<f64>) -> Self {
        Self {
            data: Rc::new(RefCell::new(AutogradData {
                grad: Array2::zeros((value.shape()[0], value.shape()[1])),
                value,
                children: Vec::new(),
                op: Op::None,
                backward: None,
            })),
        }
    }

    // method to add two Autograd objects
    pub fn add(&self, other: &Autograd) -> Autograd {
        let value = &self.data.borrow().value + &other.data.borrow().value;

        let result = Autograd::new(value);
        result.data.borrow_mut().children.push(self.clone());
        result.data.borrow_mut().children.push(other.clone());
        result.data.borrow_mut().op = Op::Add;
        result.data.borrow_mut().backward = Some(|_| {});

        result
    }

    pub fn mul(&self, other: &Autograd) -> Autograd {
        let value = &self.data.borrow().value.dot(&other.data.borrow().value);

        let result = Autograd::new(value.clone());
        result.data.borrow_mut().children.push(self.clone());
        result.data.borrow_mut().children.push(other.clone());
        result.data.borrow_mut().op = Op::Mul;
        result.data.borrow_mut().backward = Some(|_| {});

        result
    }

    pub fn tanh(&self) -> Autograd {
        let value = self.data.borrow().value.mapv(|x| x.tanh());
        let result = Autograd::new(value);

        result.data.borrow_mut().children.push(self.clone());
        result.data.borrow_mut().op = Op::Tanh;
        result.data.borrow_mut().backward = Some(|_| {});

        result
    }

    pub fn relu(&self) -> Autograd {
        let value = self.data.borrow().value.mapv(|x| x.max(0.0));
        let result = Autograd::new(value);

        result.data.borrow_mut().children.push(self.clone());
        result.data.borrow_mut().op = Op::ReLU;
        result.data.borrow_mut().backward = Some(|_| {});

        result
    }

    fn build_topo(
        &self,
        topo: &mut Vec<Autograd>,
        visited: &mut HashSet<*const RefCell<AutogradData>>,
    ) {
        let ptr = Rc::as_ptr(&self.data);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for child in &self.data.borrow().children {
                child.build_topo(topo, visited);
            }
            topo.push(self.clone());
        }
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        self.build_topo(&mut topo, &mut visited);

        for node in topo.iter().rev() {
            let data = node.data.borrow();
            if let Some(_backward_fn) = data.backward {
                let value = data.value.clone();
                let grad = data.grad.clone();
                let children = data.children.clone();
                let op = data.op;
                drop(data);

                match op {
                    Op::Add => {
                        // y = a + b -> da = dy, db = dy
                        children[0].data.borrow_mut().grad += &grad;
                        children[1].data.borrow_mut().grad += &grad;
                    }
                    Op::Mul => {
                        // y = a * b -> da = dy * b^T, db = a^T * dy
                        let v0 = children[0].data.borrow().value.clone();
                        let v1 = children[1].data.borrow().value.clone();

                        children[0].data.borrow_mut().grad += &grad.dot(&v1.t());
                        children[1].data.borrow_mut().grad += &v0.t().dot(&grad);
                    }
                    Op::Tanh => {
                        // y = tanh(x) -> dy/dx = 1 - tanh(x)^2
                        let mut v0 = children[0].data.borrow_mut();

                        let local_deriv = value.mapv(|x| 1.0 - x * x);

                        v0.grad += &(&local_deriv * &grad);
                    }
                    Op::ReLU => {
                        // y = relu(x) -> dy/dx = 1 if x > 0, 0 otherwise
                        let mut v0 = children[0].data.borrow_mut();

                        let mask = grad * value.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

                        v0.grad += &mask;
                    }
                    Op::None => {}
                }
            }
        }
    }

    pub fn zero_grad(&self) {
        let shape = {
            let data = self.data.borrow();
            (data.value.shape()[0], data.value.shape()[1])
        };
        self.data.borrow_mut().grad = Array2::zeros(shape);
    }

    pub fn value(&self) -> Array2<f64> {
        self.data.borrow().value.clone()
    }

    pub fn grad(&self) -> Array2<f64> {
        self.data.borrow().grad.clone()
    }

    pub fn set_value(&self, value: Array2<f64>) {
        self.data.borrow_mut().value = value;
    }

    pub fn set_grad(&self, grad: Array2<f64>) {
        self.data.borrow_mut().grad = grad;
    }
}

impl Clone for Autograd {
    fn clone(&self) -> Self {
        Self {
            data: Rc::clone(&self.data),
        }
    }
}

impl std::fmt::Debug for Autograd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.borrow();
        f.debug_struct("Autograd")
            .field("value", &data.value)
            .field("grad", &data.grad)
            .field("children", &data.children)
            .field("op", &data.op)
            .field("backward", &data.backward.as_ref().map(|_| "Fn"))
            .finish()
    }
}
