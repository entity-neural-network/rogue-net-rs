use std::io::Read;

use indexmap::IndexMap;
use ndarray::{Array, IxDyn};
use rmpv::Value;

#[derive(Debug, Clone)]
pub enum Error {
    ParseError(String),
}

pub fn decode_state_dict<R: Read>(mut rd: R) -> Result<TensorDict, Error> {
    let value = rmpv::decode::read_value(&mut rd).map_err(|e| Error::ParseError(e.to_string()))?;
    match value {
        Value::Map(items) => {
            let mut tensors = TensorDict::Dict(IndexMap::new());
            for (key, value) in items {
                let key = match key {
                    Value::String(s) => s.as_str().unwrap().to_string(),
                    _ => return Err(Error::ParseError("key is not string".to_string())),
                };
                log::debug!("{}", key);
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
pub enum Tensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

impl Tensor {
    pub fn to_ndarray_f32(&self) -> Array<f32, IxDyn> {
        match self {
            Tensor::F32 { shape, data } => {
                Array::from_shape_vec(shape.clone(), data.clone()).unwrap()
            }
            _ => panic!("Tensor::to_ndarray_f32: not a f32 tensor"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorDict {
    Tensor(Tensor),
    Dict(IndexMap<String, TensorDict>),
}

impl TensorDict {
    pub fn insert(&mut self, key: String, value: Tensor) {
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
                            .or_insert_with(|| TensorDict::Dict(IndexMap::new()));
                        insert(sub_dict, &path[1..], value);
                    }
                }
            }
        }
        insert(self, &key.split('.').collect::<Vec<_>>(), value);
    }

    pub fn as_dict(&self) -> &IndexMap<String, TensorDict> {
        match self {
            TensorDict::Tensor(_) => panic!("as_dict on tensor"),
            TensorDict::Dict(dict) => dict,
        }
    }

    pub fn as_tensor(&self) -> &Tensor {
        match self {
            TensorDict::Tensor(tensor) => tensor,
            TensorDict::Dict(_) => panic!("as_tensor on dict"),
        }
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
