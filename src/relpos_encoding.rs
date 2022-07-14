use std::collections::HashMap;
use std::ops::AddAssign;

use indexmap::IndexMap;
use ndarray::{concatenate, prelude::*};

use crate::config::RelposEncodingConfig;
use crate::fun::relu;
use crate::linear::Linear;
use crate::msgpack::TensorDict;
use crate::state::ObsSpace;

#[derive(Debug, Clone)]
pub struct RelposEncoding {
    position_feature_indices: IndexMap<String, Vec<usize>>,
    keys: Array2<f32>,
    values: Array2<f32>,
    value_gate_proj: Linear,
    extent: Vec<usize>,
    strides: Vec<usize>,
    scale: f32,
}

impl RelposEncoding {
    pub fn new(
        state_dict: &TensorDict,
        config: &RelposEncodingConfig,
        obs_space: &ObsSpace,
    ) -> Self {
        let rc = config;
        assert!(!rc.per_entity_values, "per_entity_values is not supported");
        assert!(
            rc.exclude_entities.is_empty(),
            "exclude_entities is not supported"
        );
        assert!(
            !rc.value_relpos_projection,
            "value_relpos_projection is not supported"
        );
        assert!(
            !rc.key_relpos_projection,
            "key_relpos_projection is not supported"
        );
        assert!(
            !rc.per_entity_projections,
            "per_entity_projections is not supported"
        );
        assert!(!rc.radial, "relpos radial is not supported");
        assert!(!rc.distance, "relpos distance is not supported");
        assert!(
            rc.rotation_vec_features.is_none(),
            "relpos rotation_vec_features is not supported"
        );
        assert!(
            rc.rotation_angle_feature.is_none(),
            "relpos rotation_angle_feature is not supported"
        );
        assert!(!rc.interpolate, "relpos interpolate is not supported");
        assert!(rc.value_gate == "relu", "only relu value_gate is supported");

        let dict = state_dict.as_dict();
        let keys = dict["keys"].as_dict()["weight"]
            .as_tensor()
            .to_ndarray_f32();
        let values = dict["values"].as_dict()["weight"]
            .as_tensor()
            .to_ndarray_f32();
        let value_gate_proj = Linear::from(&dict["value_gate_proj"]);
        let extent = config.extent.iter().map(|x| *x as usize).collect();
        let mut strides = vec![];
        let mut stride = 1;
        for e in &extent {
            strides.push(stride);
            stride *= 2 * e + 1;
        }
        // Find index corresponding to each position feature.
        let mut position_feature_indices = IndexMap::new();
        for (entity_name, entity) in &obs_space.entities {
            let mut indices = vec![];
            for feature in &config.position_features {
                if entity.features.contains(feature) {
                    indices.push(
                        entity
                            .features
                            .iter()
                            .position(|x| x == feature)
                            .unwrap_or_else(|| {
                                panic!("Feature {} not found in entity {}", feature, entity_name)
                            }),
                    );
                }
            }
            position_feature_indices.insert(entity_name.clone(), indices);
        }
        RelposEncoding {
            keys: keys.into_dimensionality().unwrap(),
            values: values.into_dimensionality().unwrap(),
            value_gate_proj,
            extent,
            strides,
            position_feature_indices,
            scale: config.scale,
        }
    }

    pub fn relpos_indices(&self, entities: &HashMap<String, Array2<f32>>) -> Array2<usize> {
        // Select position features for each entity.
        let mut poss = vec![];
        for (entity_name, indices) in &self.position_feature_indices {
            let all_features = entities.get(entity_name).unwrap();
            let mut pos_features = Array2::zeros((all_features.dim().0, indices.len()));
            for j in 0..all_features.dim().0 {
                for (k, index) in indices.iter().enumerate() {
                    pos_features[[j, k]] = all_features[[j, *index]];
                }
            }
            poss.push(pos_features);
        }
        let positions =
            concatenate(Axis(0), &poss.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();

        // Compute relative position of each entity to every other entity.
        let relative_positions =
            &positions.view().insert_axis(Axis(1)) - &positions.view().insert_axis(Axis(0));
        log::debug!("{:?}", relative_positions);

        // Convert relative position to relative position index.
        let mut relpos_indices =
            Array2::zeros((relative_positions.dim().0, relative_positions.dim().1));
        let mult = 1.0 / self.scale;
        for i in 0..relative_positions.dim().0 {
            for j in 0..relative_positions.dim().1 {
                for (k, stride) in self.strides.iter().enumerate() {
                    let index = (relative_positions[[i, j, k]] * mult)
                        .min(self.extent[k] as f32)
                        .max(-(self.extent[k] as f32));
                    relpos_indices[[i, j]] +=
                        (index + self.extent[k] as f32).round() as usize * stride;
                }
            }
        }
        log::debug!("{:?}", relpos_indices);
        relpos_indices
    }

    pub fn relattn_logits(
        &self,
        relpos_indices: &Array2<usize>,
        q: ArrayView2<f32>,
    ) -> Array2<f32> {
        // q: seq x dhead
        // relpos_indices: seq x seq
        // relative_keys: extent x dhead

        // We have a seq x dhead vector of queries:
        //
        //   q0 q1 q2 q3
        //
        // We have a seq x seq matrix of relative position indices:
        //
        //   rp00 rp01 rp02 rp03
        //   rp10 rp11 rp12 rp13
        //   rp20 rp21 rp22 rp23
        //   rp30 rp31 rp32 rp33
        //
        // We index into the relative keys to get the relative keys corresponding to each relative position:
        // key[rpxy] == kxy
        //
        //   k00 k01 k02 k03
        //   k10 k11 k12 k13
        //   k20 k21 k22 k23
        //   k30 k31 k32 k33
        //
        // We then compute the dot product of the relative keys and the queries:
        //
        //   k00.q0 k01.q0 k02.q0 k03.q0
        //   k10.q1 k11.q1 k12.q1 k13.q1
        //   k20.q2 k21.q2 k22.q2 k23.q2
        //   k30.q3 k31.q3 k32.q3 k33.q3

        let s = relpos_indices.dim().0;
        let mut relattn_logits = Array2::zeros((s, s));
        let factor = 1.0 / (self.keys.dim().1 as f32).sqrt();
        for s in 0..q.dim().0 {
            let query = q.slice(s![s, ..]);
            for t in 0..q.dim().0 {
                let key_index = relpos_indices[[s, t]];
                let key = self.keys.slice(s![key_index, ..]);
                relattn_logits[[s, t]] = (&query * &key).sum() * factor;
            }
        }
        relattn_logits
    }

    pub fn relpos_values(
        &self,
        relpos_indices: &Array2<usize>,
        attn: &Array2<f32>,
        x: ArrayView2<f32>,
    ) -> Array2<f32> {
        // TODO: duplicated work, vgate is shared between heads
        let vgate = relu(self.value_gate_proj.forward(x.view()).view());
        let mut relpos_values = Array2::zeros((attn.dim().0, self.values.dim().1));
        for s in 0..attn.dim().0 {
            for t in 0..attn.dim().1 {
                let value_index = relpos_indices[[s, t]];
                let value = self.values.slice(s![value_index, ..]);
                let gated_attn_value = attn[[s, t]] * (&value * &vgate.row(t));
                relpos_values
                    .slice_mut(s![s, ..])
                    .add_assign(&gated_attn_value);
            }
        }
        relpos_values
    }
}
