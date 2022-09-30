use ndarray::prelude::*;
use numpy::ToPyArray;
use pyo3::prelude::*;
use rogue_net::*;
use std::collections::HashMap;
use std::fs::File;

#[pyclass]
pub struct RustRogueNet(RogueNet);

#[pymethods]
impl RustRogueNet {
    #[new]
    #[args(archive = "false")]
    fn new(path: String, archive: bool) -> Self {
        if archive {
            RustRogueNet(
                RogueNet::load_archive(
                    File::open(path.clone())
                        .map_err(|e| format!("Failed to open archive {}: {}", path, e))
                        .unwrap(),
                )
                .unwrap(),
            )
        } else {
            RustRogueNet(RogueNet::load(path))
        }
    }

    fn forward(&self, py: Python, obs: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let features = obs
            .getattr(py, "features")?
            .extract::<HashMap<String, Vec<Vec<f32>>>>(py)?;
        let features = features
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    Array2::from_shape_vec(
                        (v.len(), v[0].len()),
                        v.into_iter().flatten().collect(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let actions = obs
            .getattr(py, "actions")?
            .extract::<HashMap<String, Py<PyAny>>>(py)?;
        let mut actors = vec![];
        for action in actions.values() {
            let acts = action
                .getattr(py, "actor_types")?
                .extract::<Vec<String>>(py)?;
            for a in acts {
                if !actors.contains(&a) {
                    actors.push(a);
                }
            }
        }
        let args = FwdArgs { features, actors };
        let (probs, actions) = self.0.forward(args);
        Ok((probs.to_pyarray(py), actions).into_py(py))
    }
}

#[pymodule]
fn rogue_net_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustRogueNet>()?;
    Ok(())
}
