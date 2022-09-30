use std::collections::HashMap;
use std::f32::consts::PI;
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
    orientation_vec_indices: Option<IndexMap<String, Option<[usize; 2]>>>,
    keys: Array2<f32>,
    values: Array2<f32>,
    value_gate_proj: Linear,
    extent: Vec<usize>,
    strides: Vec<usize>,
    scale: f32,
    radial: bool,
    distance: bool,
    interpolate: bool,
}

#[derive(Debug)]
pub enum RelposIndices {
    Simple(Array2<usize>),
    Interpolated(Array3<usize>, Array3<f32>),
}

// TODO: rotation_vec_features
impl RelposEncoding {
    pub fn new(
        state_dict: &TensorDict,
        config: &RelposEncodingConfig,
        obs_space: &ObsSpace,
    ) -> Self {
        let rc = config;
        assert!(!rc.per_entity_values, "per_entity_values is not supported");
        assert!(
            rc.radial == rc.distance,
            "radial currently requires distance",
        );
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
        assert!(
            rc.rotation_angle_feature.is_none(),
            "relpos rotation_angle_feature is not supported"
        );
        assert!(rc.value_gate == "relu", "only relu value_gate is supported");

        let dict = state_dict.as_dict();
        let keys = dict["keys"].as_dict()["weight"]
            .as_tensor()
            .to_ndarray_f32();
        let values = dict["values"].as_dict()["weight"]
            .as_tensor()
            .to_ndarray_f32();
        let value_gate_proj = Linear::from(&dict["value_gate_proj"]);
        let extent: Vec<usize> = config.extent.iter().map(|x| *x as usize).collect();
        let mut strides = vec![];
        if config.radial && config.distance {
            strides = vec![1, extent[0]]
        } else {
            let mut stride = 1;
            for e in &extent {
                strides.push(stride);
                stride *= 2 * e + 1;
            }
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
        // Find index corresponding to each orientation vector feature.
        let orientation_vec_indices =
            config
                .rotation_vec_features
                .as_ref()
                .map(|rotation_vec_features| {
                    assert!(
                        rotation_vec_features.len() == 2,
                        "rotation_vec_features must have length 2"
                    );
                    let mut result = IndexMap::new();
                    'outer: for (entity_name, entity) in &obs_space.entities {
                        let mut indices = vec![];
                        for feature in rotation_vec_features {
                            let index = match entity.features.iter().position(|x| x == feature) {
                                Some(index) => index,
                                None => {
                                    result.insert(entity_name.clone(), None);
                                    continue 'outer;
                                }
                            };
                            indices.push(index);
                        }
                        result.insert(entity_name.clone(), Some([indices[0], indices[1]]));
                    }
                    result
                });

        RelposEncoding {
            keys: keys.into_dimensionality().unwrap(),
            values: values.into_dimensionality().unwrap(),
            value_gate_proj,
            extent,
            strides,
            position_feature_indices,
            orientation_vec_indices,
            scale: config.scale,
            radial: config.radial,
            distance: config.distance,
            interpolate: config.interpolate,
        }
    }

    pub fn relpos_indices(&self, entities: &HashMap<String, Array2<f32>>) -> RelposIndices {
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
        let orientations = self
            .orientation_vec_indices
            .as_ref()
            .map(|orientation_vec_indices| {
                let mut orientations = Array1::zeros(positions.dim().0);
                let mut offset = 0;
                for (entity_name, indices) in orientation_vec_indices {
                    if let Some(indices) = indices {
                        for i in 0..entities[entity_name].dim().0 {
                            let x = entities[entity_name][[i, indices[0]]];
                            let y = entities[entity_name][[i, indices[1]]];
                            orientations[offset + i] = y.atan2(x);
                        }
                    }
                    offset += entities[entity_name].dim().0;
                }
                orientations
            });

        // Compute relative position of each entity to every other entity.
        let relative_positions =
            &positions.view().insert_axis(Axis(1)) - &positions.view().insert_axis(Axis(0));
        log::debug!("{:?}", relative_positions);

        if self.interpolate {
            assert!(
                self.radial && self.distance,
                "interpolate currently requires radial and distance"
            );

            let (indices, weights) = self.interpolated_polar_partition(
                relative_positions.view(),
                orientations.as_ref().map(Array1::view),
            );
            RelposIndices::Interpolated(indices, weights)
        } else {
            // Convert relative position to relative position index.
            let mut relpos_indices =
                Array2::zeros((relative_positions.dim().0, relative_positions.dim().1));
            let mult = 1.0 / self.scale;
            if self.radial && self.distance {
                // TODO: not tested
                for i in 0..relative_positions.dim().0 {
                    for j in 0..relative_positions.dim().1 {
                        let (x, y) = (relative_positions[[i, j, 0]], relative_positions[[i, j, 1]]);
                        assert!(orientations.is_none(), "not implemented");
                        let angle = y.atan2(x) + 2.0 * PI;
                        let ia = ((angle / (2.0 * PI) * self.extent[0] as f32)
                            .rem_euclid(self.extent[0] as f32))
                            as usize;
                        let distance = (x * x + y * y).sqrt();
                        let id = ((distance * mult) as usize).min(self.extent[1] - 1);
                        relpos_indices[[i, j]] = ia * self.strides[0] + id * self.strides[1];
                    }
                }
            } else {
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
            }
            log::debug!("{:?}", relpos_indices);
            RelposIndices::Simple(relpos_indices)
        }
    }

    pub fn relattn_logits(
        &self,
        relpos_indices: &RelposIndices,
        q: ArrayView2<f32>,
    ) -> Array2<f32> {
        match relpos_indices {
            RelposIndices::Simple(relpos_indices) => {
                self.single_relattn_logits(&relpos_indices.view(), q)
            }
            RelposIndices::Interpolated(relpos_indices, weights) => {
                let s = relpos_indices.dim().0;
                let mut relattn_logits: Array2<f32> = Array2::zeros((s, s));
                for i in 0..weights.dim().2 {
                    relattn_logits += &(&self
                        .single_relattn_logits(&relpos_indices.slice(s![.., .., i]), q)
                        * &weights.slice(s![.., .., i]));
                }
                relattn_logits
            }
        }
    }

    fn single_relattn_logits(
        &self,
        relpos_indices: &ArrayView2<usize>,
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
        relpos_indices: &RelposIndices,
        attn: &Array2<f32>,
        x: ArrayView2<f32>,
    ) -> Array2<f32> {
        match relpos_indices {
            RelposIndices::Simple(relpos_indices) => {
                self.single_relpos_values(relpos_indices.view(), attn, x)
            }
            RelposIndices::Interpolated(relpos_indices, weights) => {
                let s = relpos_indices.dim().0;
                let mut relpos_values: Array2<f32> = Array2::zeros((s, self.keys.dim().1));
                for i in 0..weights.dim().2 {
                    let relv = self.single_relpos_values(
                        relpos_indices.slice(s![.., .., i]),
                        &(attn * &weights.slice(s![.., .., i])),
                        x,
                    );
                    relpos_values += &relv;
                }
                relpos_values
            }
        }
    }

    fn single_relpos_values(
        &self,
        relpos_indices: ArrayView2<usize>,
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

    fn interpolated_radial_partition(
        &self,
        relative_positions: ArrayView3<f32>,
        orientations: Option<ArrayView1<f32>>,
    ) -> (Array3<usize>, Array3<f32>) {
        let mut indices =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 2));
        let mut weights =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 2));

        for i in 0..relative_positions.dim().0 {
            for j in 0..relative_positions.dim().1 {
                let (x, y) = (relative_positions[[i, j, 0]], relative_positions[[i, j, 1]]);
                let angle = y.atan2(x) - orientations.map(|o| o[i]).unwrap_or(0.0) + 2.0 * PI;
                let norm_angle =
                    (angle / (2.0 * PI) * self.extent[0] as f32).rem_euclid(self.extent[0] as f32);
                let index1 = norm_angle as usize;
                let index2 = (index1 + 1).rem_euclid(self.extent[0]);
                let weight1 = (index2 as f32 - norm_angle).rem_euclid(1.0);
                let weight2 = 1.0 - weight1;

                indices[[i, j, 0]] = index1;
                indices[[i, j, 1]] = index2;
                weights[[i, j, 0]] = weight1;
                weights[[i, j, 1]] = weight2;
            }
        }

        (indices, weights)
    }

    fn interpolated_distance_partition(
        &self,
        relative_positions: ArrayView3<f32>,
    ) -> (Array3<usize>, Array3<f32>) {
        let mut indices =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 2));
        let mut weights =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 2));

        let mult = 1.0 / self.scale;
        for i in 0..relative_positions.dim().0 {
            for j in 0..relative_positions.dim().1 {
                let (x, y) = (relative_positions[[i, j, 0]], relative_positions[[i, j, 1]]);
                let distance = (x * x + y * y).sqrt() * mult;
                let index1 = (distance as usize).min(self.extent[1] - 1);
                let index2 = (index1 + 1).min(self.extent[1] - 1);
                let weight1 = index2 as f32 - distance;
                let weight2 = 1.0 - weight1;
                // assert!(
                //     weight1 >= 0.0,
                //     "weight1: {weight1} {distance} {index1} {index2}"
                // );
                // assert!(
                //     weight2 >= 0.0,
                //     "weight2: {weight2} {distance} {index1} {index2}"
                // );
                indices[[i, j, 0]] = index1;
                indices[[i, j, 1]] = index2;
                weights[[i, j, 0]] = weight1;
                weights[[i, j, 1]] = weight2;
            }
        }

        (indices, weights)
    }

    fn interpolated_polar_partition(
        &self,
        relative_positions: ArrayView3<f32>,
        orientations: Option<ArrayView1<f32>>,
    ) -> (Array3<usize>, Array3<f32>) {
        let mut indices =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 4));
        let mut weights =
            Array3::zeros((relative_positions.dim().0, relative_positions.dim().1, 4));

        let (aindices, aweights) =
            self.interpolated_radial_partition(relative_positions, orientations);
        let (dindices, dweights) = self.interpolated_distance_partition(relative_positions);

        let indices0 = &aindices.slice(s![.., .., 0]) * self.strides[0]
            + &dindices.slice(s![.., .., 0]) * self.strides[1];
        let indices1 = &aindices.slice(s![.., .., 1]) * self.strides[0]
            + &dindices.slice(s![.., .., 0]) * self.strides[1];
        let indices2 = &aindices.slice(s![.., .., 0]) * self.strides[0]
            + &dindices.slice(s![.., .., 1]) * self.strides[1];
        let indices3 = &aindices.slice(s![.., .., 1]) * self.strides[0]
            + &dindices.slice(s![.., .., 1]) * self.strides[1];
        indices.slice_mut(s![.., .., 0]).assign(&indices0);
        indices.slice_mut(s![.., .., 1]).assign(&indices1);
        indices.slice_mut(s![.., .., 2]).assign(&indices2);
        indices.slice_mut(s![.., .., 3]).assign(&indices3);

        let weights0 = &aweights.slice(s![.., .., 0]) * &dweights.slice(s![.., .., 0]);
        let weights1 = &aweights.slice(s![.., .., 1]) * &dweights.slice(s![.., .., 0]);
        let weights2 = &aweights.slice(s![.., .., 0]) * &dweights.slice(s![.., .., 1]);
        let weights3 = &aweights.slice(s![.., .., 1]) * &dweights.slice(s![.., .., 1]);
        weights.slice_mut(s![.., .., 0]).assign(&weights0);
        weights.slice_mut(s![.., .., 1]).assign(&weights1);
        weights.slice_mut(s![.., .., 2]).assign(&weights2);
        weights.slice_mut(s![.., .., 3]).assign(&weights3);

        (indices, weights)
    }
}
