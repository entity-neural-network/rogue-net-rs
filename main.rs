use ndarray::prelude::*;
use ndarray::Array;
use std::collections::HashMap;
use std::io::Cursor;

use rmpv::{Value, ValueRef};

#[derive(Debug, Clone)]
enum Error {
    ParseError(String),
}

#[derive(Debug, Clone)]
enum Tensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

impl Tensor {
    fn to_ndarray_f32(&self) -> Array<f32, IxDyn> {
        match self {
            Tensor::F32 { shape, data } => {
                Array::from_shape_vec(shape.clone(), data.clone()).unwrap()
            }
            _ => panic!("Tensor::to_ndarray_f32: not a f32 tensor"),
        }
    }

    fn to_ndarray_i64(&self) -> Array<i64, IxDyn> {
        match self {
            Tensor::I64 { shape, data } => {
                Array::from_shape_vec(shape.clone(), data.clone()).unwrap()
            }
            _ => panic!("Tensor::to_ndarray_i64: not an i64 tensor"),
        }
    }
}

#[derive(Debug, Clone)]
enum TensorDict {
    Tensor(Tensor),
    Dict(HashMap<String, TensorDict>),
}

impl TensorDict {
    fn insert(&mut self, key: String, value: Tensor) {
        fn insert(dict: &mut TensorDict, path: &[&str], value: Tensor) {
            match dict {
                TensorDict::Tensor(_) => panic!("insertion into tensor"),
                TensorDict::Dict(dict) => {
                    if path.len() == 1 {
                        dict.insert(path[0].to_string(), TensorDict::Tensor(value));
                    } else {
                        let key = path[0];
                        let sub_dict = dict
                            .entry(key.to_string())
                            .or_insert_with(|| TensorDict::Dict(HashMap::new()));
                        insert(sub_dict, &path[1..], value);
                    }
                }
            }
        }
        insert(self, &key.split('.').collect::<Vec<_>>(), value);
    }

    fn as_dict(&self) -> &HashMap<String, TensorDict> {
        match self {
            TensorDict::Tensor(_) => panic!("as_dict on tensor"),
            TensorDict::Dict(dict) => dict,
        }
    }

    fn as_tensor(&self) -> &Tensor {
        match self {
            TensorDict::Tensor(tensor) => tensor,
            TensorDict::Dict(_) => panic!("as_tensor on dict"),
        }
    }
}

fn main() {
    //let bytes = include_bytes!("test.msgpack");
    let bytes = include_bytes!("checkpoints/latest-step000000024576/state.agent.msgpack");
    let state_dict = decode_state_dict(bytes).unwrap();
    println!("{:#?}", state_dict);
    let rogue_net = RogueNet::from(&state_dict);
    println!("{:#?}", rogue_net);
    rogue_net.forward(array![[3.0, 5.0], [8.0, 4.0]]);
}

fn decode_state_dict(bytes: &[u8]) -> Result<TensorDict, Error> {
    let mut cursor = Cursor::new(bytes);
    let value =
        rmpv::decode::read_value(&mut cursor).map_err(|e| Error::ParseError(e.to_string()))?;
    match value {
        Value::Map(items) => {
            let mut tensors = TensorDict::Dict(HashMap::new());
            for (key, value) in items {
                let key = match key {
                    Value::String(s) => s.as_str().unwrap().to_string(),
                    _ => return Err(Error::ParseError("key is not string".to_string())),
                };
                let tensor = decode_tensor(value)?;
                tensors.insert(key, tensor);
            }
            Ok(tensors)
        }
        _ => Err(Error::ParseError(
            "Malformed snapshot, expected top level map".to_string(),
        )),
    }
}

#[derive(Debug, Clone)]
struct RogueNet {
    embeddings: HashMap<String, Embedding>,
}

impl RogueNet {
    fn forward(&self, food: Array2<f32>) -> Array2<f32> {
        self.embeddings["Food"].forward(food)
    }
}

impl<'a> From<&'a TensorDict> for RogueNet {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let mut embeddings = HashMap::new();
        for (key, value) in dict["embedding"].as_dict()["embeddings"].as_dict() {
            let embedding = value.into();
            embeddings.insert(key.clone(), embedding);
        }

        RogueNet { embeddings }
    }
}

#[derive(Debug, Clone)]
struct Embedding {
    mean: Array<f32, Ix1>,
    std: Array<f32, Ix1>,
    proj: Linear,
    ln: LayerNorm,
}

impl<'a> From<&'a TensorDict> for Embedding {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let norm = dict["0"].as_dict();
        let mean = norm["mean"].as_tensor().to_ndarray_f32();
        let count = norm["count"].as_tensor().to_ndarray_f32();
        let squares_sum = norm["squares_sum"].as_tensor().to_ndarray_f32();
        Embedding {
            std: (squares_sum / (count - 1.0))
                .mapv(|x| if x == 0.0 { 1.0 } else { x.sqrt() })
                .into_dimensionality()
                .unwrap(),
            mean: mean.into_dimensionality().unwrap(),
            proj: Linear::from(&dict["1"]),
            ln: LayerNorm::from(&dict["3"]),
        }
    }
}

fn relu(x: ArrayView2<f32>) -> Array2<f32> {
    x.mapv(|x| if x < 0.0 { 0.0 } else { x })
}

fn clip(x: ArrayView2<f32>, min: f32, max: f32) -> Array2<f32> {
    x.mapv(|x| {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    })
}

impl Embedding {
    fn forward(&self, mut x: Array2<f32>) -> Array2<f32> {
        x = (x - &self.mean) / &self.std;
        x = clip(x.view(), -5.0, 5.0);
        x = self.proj.forward(x);
        x = relu(x.view());
        self.ln.forward(x)
    }
}

#[derive(Debug, Clone)]
struct Linear {
    weight: Array<f32, Ix2>,
    bias: Array<f32, Ix2>,
}

impl<'a> From<&'a TensorDict> for Linear {
    fn from(state_dict: &TensorDict) -> Self {
        println!("SD: {:?}", state_dict);
        let dict = state_dict.as_dict();
        let weight = dict["weight"].as_tensor().to_ndarray_f32();
        let bias = dict["bias"].as_tensor().to_ndarray_f32();
        println!("{:?}", weight);
        Linear {
            weight: weight.reversed_axes().into_dimensionality().unwrap(),
            bias: bias.insert_axis(Axis(0)).into_dimensionality().unwrap(),
        }
    }
}

impl Linear {
    fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        println!(
            "x.shape: {:?} self.weight.shape: {:?} self.bias.shape: {:?}",
            x.shape(),
            self.weight.shape(),
            self.bias.shape()
        );
        x.dot(&self.weight) + &self.bias
    }
}

#[derive(Debug, Clone)]
struct LayerNorm {
    weight: Array<f32, Ix1>,
    bias: Array<f32, Ix1>,
}

impl<'a> From<&'a TensorDict> for LayerNorm {
    fn from(state_dict: &TensorDict) -> Self {
        let dict = state_dict.as_dict();
        let weight = dict["weight"].as_tensor().to_ndarray_f32();
        let bias = dict["bias"].as_tensor().to_ndarray_f32();
        LayerNorm {
            weight: weight.into_dimensionality().unwrap(),
            bias: bias.into_dimensionality().unwrap(),
        }
    }
}

impl LayerNorm {
    fn forward(&self, x: Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let std = (&x - &mean).std_axis(Axis(1), 0.0).insert_axis(Axis(1));
        // print all shapes
        println!(
            "mean.shape: {:?} std.shape: {:?} x.shape: {:?} weight.shape: {:?} bias.shape: {:?}",
            mean.shape(),
            std.shape(),
            x.shape(),
            self.weight.shape(),
            self.bias.shape()
        );
        (x - mean) / (std + 1e-5) * &self.weight + &self.bias
    }
}

fn decode_tensor(value: Value) -> Result<Tensor, Error> {
    let items = value.as_map().unwrap();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0].0.as_slice().unwrap(), b"__tensor__");
    assert_eq!(items[0].1.as_str().unwrap(), "torch");
    assert_eq!(items[1].0.as_slice().unwrap(), b"dtype");
    assert_eq!(items[2].0.as_slice().unwrap(), b"shape");
    let shape = items[2]
        .1
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as usize)
        .collect::<Vec<_>>();
    assert_eq!(items[3].0.as_slice().unwrap(), b"data");
    let data = items[3].1.as_slice().unwrap();
    let tensor = match items[1].1.as_str().unwrap() {
        "<f4" => Tensor::F32 {
            shape,
            data: data
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>(),
        },
        "<i8" => Tensor::I64 {
            shape,
            data: data
                .chunks(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect::<Vec<_>>(),
        },
        dtype => return Err(Error::ParseError(format!("Unsupported dtype: {}", dtype))),
    };
    Ok(tensor)
}
